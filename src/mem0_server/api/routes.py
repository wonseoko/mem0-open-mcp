"""
REST API routes for mem0-server.

Provides endpoints for:
- Configuration management (CRUD)
- Memory operations (list, search, get, delete)
- Server status and health
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from mem0_server.config import (
    ConfigLoader,
    EmbedderConfig,
    EmbedderProvider,
    EmbedderProviderType,
    LLMConfig,
    LLMProvider,
    LLMProviderType,
    Mem0ServerConfig,
    OpenMemoryConfig,
    VectorStoreConfig,
    VectorStoreProvider,
    VectorStoreProviderType,
)
from mem0_server.config.schema import ServerConfig

logger = logging.getLogger(__name__)

# =============================================================================
# Request/Response Models
# =============================================================================

class ConfigResponse(BaseModel):
    """Configuration response model."""
    server: ServerConfig
    llm: LLMProvider
    embedder: EmbedderProvider
    vector_store: VectorStoreProvider
    openmemory: OpenMemoryConfig
    config_path: str | None = None


class ConfigUpdateRequest(BaseModel):
    """Configuration update request (partial update)."""
    server: ServerConfig | None = None
    llm: LLMProvider | None = None
    embedder: EmbedderProvider | None = None
    vector_store: VectorStoreProvider | None = None
    openmemory: OpenMemoryConfig | None = None


class LLMConfigUpdateRequest(BaseModel):
    """LLM configuration update request."""
    provider: LLMProviderType | None = None
    config: LLMConfig | None = None


class EmbedderConfigUpdateRequest(BaseModel):
    """Embedder configuration update request."""
    provider: EmbedderProviderType | None = None
    config: EmbedderConfig | None = None


class VectorStoreConfigUpdateRequest(BaseModel):
    """Vector store configuration update request."""
    provider: VectorStoreProviderType | None = None
    config: VectorStoreConfig | None = None


class MemoryItem(BaseModel):
    """Memory item model."""
    id: str
    memory: str
    user_id: str | None = None
    hash: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None


class MemoryListResponse(BaseModel):
    """Memory list response."""
    results: list[MemoryItem]
    count: int
    user_id: str


class MemorySearchRequest(BaseModel):
    """Memory search request."""
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    user_id: str | None = None


class MemorySearchResponse(BaseModel):
    """Memory search response."""
    results: list[MemoryItem]
    count: int
    query: str


class MemoryAddRequest(BaseModel):
    """Memory add request."""
    text: str
    user_id: str | None = None
    metadata: dict[str, Any] | None = None


class MemoryAddResponse(BaseModel):
    """Memory add response."""
    results: list[dict[str, Any]]
    count: int


class MemoryDeleteRequest(BaseModel):
    """Memory delete request."""
    memory_ids: list[str]


class MemoryDeleteResponse(BaseModel):
    """Memory delete response."""
    deleted: list[str]
    count: int


class StatusResponse(BaseModel):
    """Server status response."""
    status: str
    version: str
    config_loaded: bool
    config_path: str | None = None
    memory_available: bool
    user_id: str
    providers: dict[str, str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str


class ProvidersResponse(BaseModel):
    """Available providers response."""
    llm_providers: list[str]
    embedder_providers: list[str]
    vector_store_providers: list[str]


# =============================================================================
# API Router Factory
# =============================================================================

def create_api_router(
    config: Mem0ServerConfig,
    config_loader: ConfigLoader,
    get_memory_client_func,
) -> APIRouter:
    """Create the API router with all endpoints.
    
    Args:
        config: The server configuration.
        config_loader: The config loader for saving changes.
        get_memory_client_func: Function to get the memory client.
    
    Returns:
        FastAPI APIRouter with all endpoints configured.
    """
    router = APIRouter(prefix="/api/v1", tags=["api"])
    
    # Store mutable config reference
    current_config = {"config": config}
    
    # =========================================================================
    # Health & Status
    # =========================================================================
    
    @router.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(status="ok", service="mem0-server")
    
    @router.get("/status", response_model=StatusResponse)
    async def get_status():
        """Get server status and configuration summary."""
        from mem0_server import __version__
        
        cfg = current_config["config"]
        memory_client = await get_memory_client_func()
        
        return StatusResponse(
            status="running",
            version=__version__,
            config_loaded=config_loader.config_path is not None,
            config_path=str(config_loader.config_path) if config_loader.config_path else None,
            memory_available=memory_client is not None,
            user_id=cfg.server.user_id,
            providers={
                "llm": cfg.llm.provider.value,
                "embedder": cfg.embedder.provider.value,
                "vector_store": cfg.vector_store.provider.value,
            },
        )
    
    @router.get("/providers", response_model=ProvidersResponse)
    async def get_available_providers():
        """Get list of all available providers."""
        return ProvidersResponse(
            llm_providers=[p.value for p in LLMProviderType],
            embedder_providers=[p.value for p in EmbedderProviderType],
            vector_store_providers=[p.value for p in VectorStoreProviderType],
        )
    
    # =========================================================================
    # Configuration Management
    # =========================================================================
    
    @router.get("/config", response_model=ConfigResponse)
    async def get_configuration():
        """Get the current configuration."""
        cfg = current_config["config"]
        return ConfigResponse(
            server=cfg.server,
            llm=cfg.llm,
            embedder=cfg.embedder,
            vector_store=cfg.vector_store,
            openmemory=cfg.openmemory,
            config_path=str(config_loader.config_path) if config_loader.config_path else None,
        )
    
    @router.put("/config", response_model=ConfigResponse)
    async def update_configuration(update: ConfigUpdateRequest):
        """Update the entire configuration."""
        cfg = current_config["config"]
        
        # Apply updates
        if update.server:
            cfg.server = update.server
        if update.llm:
            cfg.llm = update.llm
        if update.embedder:
            cfg.embedder = update.embedder
        if update.vector_store:
            cfg.vector_store = update.vector_store
        if update.openmemory:
            cfg.openmemory = update.openmemory
        
        # Save to file if we have a config path
        if config_loader.config_path:
            try:
                config_loader.save(cfg, config_loader.config_path)
            except Exception as e:
                logger.warning(f"Failed to save config: {e}")
        
        current_config["config"] = cfg
        
        return ConfigResponse(
            server=cfg.server,
            llm=cfg.llm,
            embedder=cfg.embedder,
            vector_store=cfg.vector_store,
            openmemory=cfg.openmemory,
            config_path=str(config_loader.config_path) if config_loader.config_path else None,
        )
    
    @router.patch("/config", response_model=ConfigResponse)
    async def patch_configuration(update: ConfigUpdateRequest):
        """Partially update the configuration."""
        return await update_configuration(update)
    
    @router.post("/config/reset", response_model=ConfigResponse)
    async def reset_configuration():
        """Reset configuration to defaults."""
        from mem0_server.config.schema import get_default_config
        
        cfg = get_default_config()
        current_config["config"] = cfg
        
        # Save to file if we have a config path
        if config_loader.config_path:
            try:
                config_loader.save(cfg, config_loader.config_path)
            except Exception as e:
                logger.warning(f"Failed to save config: {e}")
        
        return ConfigResponse(
            server=cfg.server,
            llm=cfg.llm,
            embedder=cfg.embedder,
            vector_store=cfg.vector_store,
            openmemory=cfg.openmemory,
            config_path=str(config_loader.config_path) if config_loader.config_path else None,
        )
    
    # Individual config section endpoints
    @router.get("/config/llm", response_model=LLMProvider)
    async def get_llm_config():
        """Get LLM configuration."""
        return current_config["config"].llm
    
    @router.put("/config/llm", response_model=LLMProvider)
    async def update_llm_config(update: LLMConfigUpdateRequest):
        """Update LLM configuration."""
        cfg = current_config["config"]
        
        if update.provider:
            cfg.llm.provider = update.provider
        if update.config:
            cfg.llm.config = update.config
        
        if config_loader.config_path:
            config_loader.save(cfg, config_loader.config_path)
        
        return cfg.llm
    
    @router.get("/config/embedder", response_model=EmbedderProvider)
    async def get_embedder_config():
        """Get embedder configuration."""
        return current_config["config"].embedder
    
    @router.put("/config/embedder", response_model=EmbedderProvider)
    async def update_embedder_config(update: EmbedderConfigUpdateRequest):
        """Update embedder configuration."""
        cfg = current_config["config"]
        
        if update.provider:
            cfg.embedder.provider = update.provider
        if update.config:
            cfg.embedder.config = update.config
        
        if config_loader.config_path:
            config_loader.save(cfg, config_loader.config_path)
        
        return cfg.embedder
    
    @router.get("/config/vector_store", response_model=VectorStoreProvider)
    async def get_vector_store_config():
        """Get vector store configuration."""
        return current_config["config"].vector_store
    
    @router.put("/config/vector_store", response_model=VectorStoreProvider)
    async def update_vector_store_config(update: VectorStoreConfigUpdateRequest):
        """Update vector store configuration."""
        cfg = current_config["config"]
        
        if update.provider:
            cfg.vector_store.provider = update.provider
        if update.config:
            cfg.vector_store.config = update.config
        
        if config_loader.config_path:
            config_loader.save(cfg, config_loader.config_path)
        
        return cfg.vector_store
    
    @router.get("/config/openmemory", response_model=OpenMemoryConfig)
    async def get_openmemory_config():
        """Get OpenMemory configuration."""
        return current_config["config"].openmemory
    
    @router.put("/config/openmemory", response_model=OpenMemoryConfig)
    async def update_openmemory_config(update: OpenMemoryConfig):
        """Update OpenMemory configuration."""
        cfg = current_config["config"]
        cfg.openmemory = update
        
        if config_loader.config_path:
            config_loader.save(cfg, config_loader.config_path)
        
        return cfg.openmemory
    
    # =========================================================================
    # Memory Operations
    # =========================================================================
    
    @router.get("/memories", response_model=MemoryListResponse)
    async def list_memories(
        user_id: str | None = Query(None, description="User ID to filter memories"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum number of memories to return"),
    ):
        """List all memories for a user."""
        cfg = current_config["config"]
        uid = user_id or cfg.server.user_id
        
        memory_client = await get_memory_client_func()
        if not memory_client:
            raise HTTPException(status_code=503, detail="Memory system unavailable")
        
        try:
            memories = await memory_client.get_all(user_id=uid)
            
            results = []
            if isinstance(memories, dict) and "results" in memories:
                for m in memories["results"][:limit]:
                    results.append(MemoryItem(
                        id=m.get("id", ""),
                        memory=m.get("memory", ""),
                        user_id=m.get("user_id"),
                        hash=m.get("hash"),
                        metadata=m.get("metadata"),
                        created_at=m.get("created_at"),
                        updated_at=m.get("updated_at"),
                    ))
            elif isinstance(memories, list):
                for m in memories[:limit]:
                    results.append(MemoryItem(
                        id=m.get("id", ""),
                        memory=m.get("memory", ""),
                        user_id=m.get("user_id"),
                        hash=m.get("hash"),
                        metadata=m.get("metadata"),
                        created_at=m.get("created_at"),
                        updated_at=m.get("updated_at"),
                    ))
            
            return MemoryListResponse(
                results=results,
                count=len(results),
                user_id=uid,
            )
        except Exception as e:
            logger.exception(f"Error listing memories: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from None
    
    @router.post("/memories/search", response_model=MemorySearchResponse)
    async def search_memories(request: MemorySearchRequest):
        """Search memories by query."""
        cfg = current_config["config"]
        uid = request.user_id or cfg.server.user_id
        
        memory_client = await get_memory_client_func()
        if not memory_client:
            raise HTTPException(status_code=503, detail="Memory system unavailable")
        
        try:
            results_raw = await memory_client.search(
                query=request.query,
                user_id=uid,
                limit=request.limit,
            )
            
            results = []
            items = results_raw.get("results", []) if isinstance(results_raw, dict) else results_raw
            
            for m in items:
                results.append(MemoryItem(
                    id=m.get("id", ""),
                    memory=m.get("memory", ""),
                    user_id=m.get("user_id"),
                    hash=m.get("hash"),
                    metadata=m.get("metadata"),
                    created_at=m.get("created_at"),
                    updated_at=m.get("updated_at"),
                ))
            
            return MemorySearchResponse(
                results=results,
                count=len(results),
                query=request.query,
            )
        except Exception as e:
            logger.exception(f"Error searching memories: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from None
    
    @router.get("/memories/{memory_id}", response_model=MemoryItem)
    async def get_memory(memory_id: str):
        """Get a specific memory by ID."""
        memory_client = await get_memory_client_func()
        if not memory_client:
            raise HTTPException(status_code=503, detail="Memory system unavailable")
        
        try:
            memory = await memory_client.get(memory_id)
            
            if not memory:
                raise HTTPException(status_code=404, detail="Memory not found")
            
            return MemoryItem(
                id=memory.get("id", memory_id),
                memory=memory.get("memory", ""),
                user_id=memory.get("user_id"),
                hash=memory.get("hash"),
                metadata=memory.get("metadata"),
                created_at=memory.get("created_at"),
                updated_at=memory.get("updated_at"),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error getting memory: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from None
    
    @router.post("/memories", response_model=MemoryAddResponse)
    async def add_memory(request: MemoryAddRequest):
        """Add a new memory."""
        cfg = current_config["config"]
        uid = request.user_id or cfg.server.user_id
        
        memory_client = await get_memory_client_func()
        if not memory_client:
            raise HTTPException(status_code=503, detail="Memory system unavailable")
        
        try:
            result = await memory_client.add(
                request.text,
                user_id=uid,
                metadata=request.metadata or {},
            )
            
            results = result.get("results", []) if isinstance(result, dict) else []
            
            return MemoryAddResponse(
                results=results,
                count=len(results),
            )
        except Exception as e:
            logger.exception(f"Error adding memory: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from None
    
    @router.delete("/memories/{memory_id}")
    async def delete_memory(memory_id: str):
        """Delete a specific memory."""
        memory_client = await get_memory_client_func()
        if not memory_client:
            raise HTTPException(status_code=503, detail="Memory system unavailable")
        
        try:
            await memory_client.delete(memory_id)
            return {"deleted": memory_id, "status": "ok"}
        except Exception as e:
            logger.exception(f"Error deleting memory: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from None
    
    @router.post("/memories/delete", response_model=MemoryDeleteResponse)
    async def delete_memories(request: MemoryDeleteRequest):
        """Delete multiple memories by IDs."""
        memory_client = await get_memory_client_func()
        if not memory_client:
            raise HTTPException(status_code=503, detail="Memory system unavailable")
        
        deleted = []
        for memory_id in request.memory_ids:
            try:
                await memory_client.delete(memory_id)
                deleted.append(memory_id)
            except Exception as e:
                logger.warning(f"Failed to delete memory {memory_id}: {e}")
        
        return MemoryDeleteResponse(
            deleted=deleted,
            count=len(deleted),
        )
    
    @router.delete("/memories")
    async def delete_all_memories(
        user_id: str | None = Query(None, description="User ID to delete memories for"),
        confirm: bool = Query(False, description="Confirm deletion"),
    ):
        """Delete all memories for a user."""
        if not confirm:
            raise HTTPException(
                status_code=400, 
                detail="Set confirm=true to delete all memories"
            )
        
        cfg = current_config["config"]
        uid = user_id or cfg.server.user_id
        
        memory_client = await get_memory_client_func()
        if not memory_client:
            raise HTTPException(status_code=503, detail="Memory system unavailable")
        
        try:
            await memory_client.delete_all(user_id=uid)
            return {"status": "ok", "message": f"All memories deleted for user {uid}"}
        except Exception as e:
            logger.exception(f"Error deleting all memories: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from None
    
    @router.get("/memories/stats/summary")
    async def get_memory_stats(
        user_id: str | None = Query(None, description="User ID to get stats for"),
    ):
        """Get memory statistics."""
        cfg = current_config["config"]
        uid = user_id or cfg.server.user_id
        
        memory_client = await get_memory_client_func()
        if not memory_client:
            return {
                "status": "unavailable",
                "message": "Memory system not initialized",
                "user_id": uid,
                "memory_count": 0,
            }
        
        try:
            memories = await memory_client.get_all(user_id=uid)
            
            if isinstance(memories, dict) and "results" in memories:
                count = len(memories["results"])
            elif isinstance(memories, list):
                count = len(memories)
            else:
                count = 0
            
            return {
                "status": "ok",
                "user_id": uid,
                "memory_count": count,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "user_id": uid,
                "memory_count": 0,
            }
    
    return router
