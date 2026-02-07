"""
MCP Server implementation for mem0-server.

This module provides a standalone MCP server that can run without Docker or
the full OpenMemory API. It uses the mem0 library directly for memory operations.
"""

from __future__ import annotations

import contextvars
import json
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport

from mem0_server.config import ConfigLoader, Mem0ServerConfig

logger = logging.getLogger(__name__)

# Global config loader reference for API routes
_config_loader: ConfigLoader | None = None

# Context variables for user_id and client_name
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id")
client_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("client_name")


class MCPServerManager:
    """Manages the MCP server and mem0 memory client."""
    
    def __init__(self, config: Mem0ServerConfig):
        self.config = config
        self._memory_client = None
        self._memory_client_initialized = False
        self._mcp = FastMCP("mem0-mcp-server")
        self._router = APIRouter(prefix="/mcp")
        self._sse = SseServerTransport("/mcp/messages/")
        self._setup_tools()
        self._setup_routes()
    
    @property
    def mcp(self) -> FastMCP:
        """Get the FastMCP instance."""
        return self._mcp
    
    @property
    def router(self) -> APIRouter:
        """Get the API router with MCP endpoints."""
        return self._router
    
    async def get_memory_client(self):
        """Get or initialize the mem0 async memory client."""
        if self._memory_client_initialized:
            return self._memory_client
        
        try:
            from mem0 import AsyncMemory
            
            mem0_config = self.config.to_mem0_config()
            logger.info(f"Initializing mem0 AsyncMemory with config: {json.dumps(mem0_config, indent=2)}")
            
            self._memory_client = await AsyncMemory.from_config(mem0_config)
            self._memory_client_initialized = True
            logger.info("mem0 AsyncMemory client initialized successfully")
            return self._memory_client
            
        except Exception as e:
            logger.error(f"Failed to initialize mem0 async client: {e}")
            raise
    
    async def get_memory_client_safe(self):
        """Get memory client with error handling. Returns None if unavailable."""
        try:
            return await self.get_memory_client()
        except Exception as e:
            logger.warning(f"Memory client unavailable: {e}")
            return None
    
    def _setup_tools(self) -> None:
        """Register MCP tools for memory operations."""
        
        @self._mcp.tool(
            description="Add a new memory. Called when user shares information about themselves, "
                       "preferences, or anything relevant for future conversations. "
                       "Also called when user explicitly asks to remember something."
        )
        async def add_memories(text: str) -> str:
            uid = user_id_var.get(None) or self.config.server.user_id
            client_name = client_name_var.get(None) or "default"
            
            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                return "Error: Memory system is currently unavailable. Please try again later."
            
            try:
                response = await memory_client.add(
                    text,
                    user_id=uid,
                    metadata={
                        "source_app": "mem0-server",
                        "mcp_client": client_name,
                    }
                )
                return json.dumps(response, indent=2)
            except Exception as e:
                logger.exception(f"Error adding memory: {e}")
                return f"Error adding to memory: {e}"
        
        @self._mcp.tool(
            description="Search through stored memories. Called when user asks anything "
                       "to find relevant context from past conversations."
        )
        async def search_memory(query: str, limit: int = 10) -> str:
            uid = user_id_var.get(None) or self.config.server.user_id
            
            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                return "Error: Memory system is currently unavailable. Please try again later."
            
            try:
                results = await memory_client.search(query=query, user_id=uid, limit=limit)
                
                if not results or not results.get("results"):
                    return json.dumps({"results": [], "message": "No relevant memories found."})
                
                return json.dumps(results, indent=2)
            except Exception as e:
                logger.exception(f"Error searching memory: {e}")
                return f"Error searching memory: {e}"
        
        @self._mcp.tool(description="List all memories for the user.")
        async def list_memories() -> str:
            uid = user_id_var.get(None) or self.config.server.user_id
            
            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                return "Error: Memory system is currently unavailable. Please try again later."
            
            try:
                memories = await memory_client.get_all(user_id=uid)
                return json.dumps(memories, indent=2)
            except Exception as e:
                logger.exception(f"Error listing memories: {e}")
                return f"Error getting memories: {e}"
        
        @self._mcp.tool(description="Get a specific memory by ID.")
        async def get_memory(memory_id: str) -> str:
            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                return "Error: Memory system is currently unavailable. Please try again later."
            
            try:
                memory = await memory_client.get(memory_id)
                return json.dumps(memory, indent=2)
            except Exception as e:
                logger.exception(f"Error getting memory: {e}")
                return f"Error getting memory: {e}"
        
        @self._mcp.tool(description="Delete specific memories by their IDs.")
        async def delete_memories(memory_ids: list[str]) -> str:
            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                return "Error: Memory system is currently unavailable. Please try again later."
            
            try:
                deleted = []
                for memory_id in memory_ids:
                    try:
                        await memory_client.delete(memory_id)
                        deleted.append(memory_id)
                    except Exception as e:
                        logger.warning(f"Failed to delete memory {memory_id}: {e}")
                
                return f"Successfully deleted {len(deleted)} memories"
            except Exception as e:
                logger.exception(f"Error deleting memories: {e}")
                return f"Error deleting memories: {e}"
        
        @self._mcp.tool(description="Delete all memories for the user.")
        async def delete_all_memories() -> str:
            uid = user_id_var.get(None) or self.config.server.user_id
            
            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                return "Error: Memory system is currently unavailable. Please try again later."
            
            try:
                await memory_client.delete_all(user_id=uid)
                return "Successfully deleted all memories"
            except Exception as e:
                logger.exception(f"Error deleting memories: {e}")
                return f"Error deleting memories: {e}"
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes for MCP endpoints."""
        
        @self._router.get("/{client_name}/sse/{user_id}")
        async def handle_sse(request: Request) -> None:
            """Handle SSE connections for a specific user and client."""
            uid = request.path_params.get("user_id", self.config.server.user_id)
            user_token = user_id_var.set(uid)
            client_name = request.path_params.get("client_name", "default")
            client_token = client_name_var.set(client_name)
            
            try:
                async with self._sse.connect_sse(
                    request.scope,
                    request.receive,
                    request._send,
                ) as (read_stream, write_stream):
                    await self._mcp._mcp_server.run(
                        read_stream,
                        write_stream,
                        self._mcp._mcp_server.create_initialization_options(),
                    )
            finally:
                user_id_var.reset(user_token)
                client_name_var.reset(client_token)
        
        @self._router.post("/messages/")
        async def handle_post_message(request: Request) -> dict[str, str]:
            """Handle POST messages for SSE."""
            try:
                body = await request.body()
                
                async def receive():
                    return {"type": "http.request", "body": body, "more_body": False}
                
                async def send(message):
                    return {}
                
                await self._sse.handle_post_message(request.scope, receive, send)
                return {"status": "ok"}
            except Exception as e:
                logger.exception(f"Error handling post message: {e}")
                return {"status": "error", "message": str(e)}
        
        @self._router.post("/{client_name}/sse/{user_id}/messages/")
        async def handle_client_post_message(request: Request) -> dict[str, str]:
            """Handle POST messages for specific client/user."""
            return await handle_post_message(request)


class MCPServerManagerStdio:
    """Manages the MCP server for stdio transport without FastAPI/HTTP infrastructure."""
    
    def __init__(self, config: Mem0ServerConfig):
        self.config = config
        self._memory_client = None
        self._memory_client_initialized = False
        self._mcp = FastMCP("mem0-mcp-server")
        self._setup_tools()
    
    @property
    def mcp(self) -> FastMCP:
        """Get the FastMCP instance."""
        return self._mcp
    
    async def get_memory_client(self):
        """Get or initialize the mem0 async memory client."""
        if self._memory_client_initialized:
            return self._memory_client
        
        try:
            from mem0 import AsyncMemory
            
            mem0_config = self.config.to_mem0_config()
            logger.info(f"Initializing mem0 AsyncMemory with config: {json.dumps(mem0_config, indent=2)}")
            
            self._memory_client = await AsyncMemory.from_config(mem0_config)
            self._memory_client_initialized = True
            logger.info("mem0 AsyncMemory client initialized successfully")
            return self._memory_client
            
        except Exception as e:
            logger.error(f"Failed to initialize mem0 async client: {e}")
            raise
    
    async def get_memory_client_safe(self):
        """Get memory client with error handling. Returns None if unavailable."""
        try:
            return await self.get_memory_client()
        except Exception as e:
            logger.warning(f"Memory client unavailable: {e}")
            return None
    
    def _setup_tools(self) -> None:
        """Register MCP tools for memory operations."""
        
        @self._mcp.tool(
            description="Add a new memory. Called when user shares information about themselves, "
                       "preferences, or anything relevant for future conversations. "
                       "Also called when user explicitly asks to remember something."
        )
        async def add_memories(text: str) -> str:
            uid = user_id_var.get(None) or self.config.server.user_id
            client_name = client_name_var.get(None) or "default"
            
            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                return "Error: Memory system is currently unavailable. Please try again later."
            
            try:
                response = await memory_client.add(
                    text,
                    user_id=uid,
                    metadata={
                        "source_app": "mem0-server",
                        "mcp_client": client_name,
                    }
                )
                return json.dumps(response, indent=2)
            except Exception as e:
                logger.exception(f"Error adding memory: {e}")
                return f"Error adding to memory: {e}"
        
        @self._mcp.tool(
            description="Search through stored memories. Called when user asks anything "
                       "to find relevant context from past conversations."
        )
        async def search_memory(query: str, limit: int = 10) -> str:
            uid = user_id_var.get(None) or self.config.server.user_id
            
            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                return "Error: Memory system is currently unavailable. Please try again later."
            
            try:
                results = await memory_client.search(query=query, user_id=uid, limit=limit)
                
                if not results or not results.get("results"):
                    return json.dumps({"results": [], "message": "No relevant memories found."})
                
                return json.dumps(results, indent=2)
            except Exception as e:
                logger.exception(f"Error searching memory: {e}")
                return f"Error searching memory: {e}"
        
        @self._mcp.tool(description="List all memories for the user.")
        async def list_memories() -> str:
            uid = user_id_var.get(None) or self.config.server.user_id
            
            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                return "Error: Memory system is currently unavailable. Please try again later."
            
            try:
                memories = await memory_client.get_all(user_id=uid)
                return json.dumps(memories, indent=2)
            except Exception as e:
                logger.exception(f"Error listing memories: {e}")
                return f"Error getting memories: {e}"
        
        @self._mcp.tool(description="Get a specific memory by ID.")
        async def get_memory(memory_id: str) -> str:
            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                return "Error: Memory system is currently unavailable. Please try again later."
            
            try:
                memory = await memory_client.get(memory_id)
                return json.dumps(memory, indent=2)
            except Exception as e:
                logger.exception(f"Error getting memory: {e}")
                return f"Error getting memory: {e}"
        
        @self._mcp.tool(description="Delete specific memories by their IDs.")
        async def delete_memories(memory_ids: list[str]) -> str:
            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                return "Error: Memory system is currently unavailable. Please try again later."
            
            try:
                deleted = []
                for memory_id in memory_ids:
                    try:
                        await memory_client.delete(memory_id)
                        deleted.append(memory_id)
                    except Exception as e:
                        logger.warning(f"Failed to delete memory {memory_id}: {e}")
                
                return f"Successfully deleted {len(deleted)} memories"
            except Exception as e:
                logger.exception(f"Error deleting memories: {e}")
                return f"Error deleting memories: {e}"
        
        @self._mcp.tool(description="Delete all memories for the user.")
        async def delete_all_memories() -> str:
            uid = user_id_var.get(None) or self.config.server.user_id
            
            memory_client = await self.get_memory_client_safe()
            if not memory_client:
                return "Error: Memory system is currently unavailable. Please try again later."
            
            try:
                await memory_client.delete_all(user_id=uid)
                return "Successfully deleted all memories"
            except Exception as e:
                logger.exception(f"Error deleting memories: {e}")
                return f"Error deleting memories: {e}"


def create_mcp_manager(config: Mem0ServerConfig) -> MCPServerManager:
     """Create an MCPServerManager instance without HTTP server.
     
     This function is used for stdio mode which only needs the MCP protocol layer
     without the FastAPI HTTP server infrastructure.
     
     Args:
         config: The server configuration.
     
     Returns:
         Configured MCPServerManager instance.
     """
     return MCPServerManager(config)


def create_mcp_manager_stdio(config: Mem0ServerConfig) -> MCPServerManagerStdio:
    """Create an MCPServerManagerStdio instance for stdio transport.
    
    This function creates an MCP manager optimized for stdio transport without
    any FastAPI HTTP infrastructure. It only manages the FastMCP instance and tools.
    
    Args:
        config: The server configuration.
    
    Returns:
        Configured MCPServerManagerStdio instance.
    """
    return MCPServerManagerStdio(config)


def create_app(config: Mem0ServerConfig, config_loader: ConfigLoader | None = None) -> FastAPI:
    """Create the FastAPI application with MCP server and REST API.
    
    Args:
        config: The server configuration.
        config_loader: Optional config loader for saving configuration changes.
    
    Returns:
        Configured FastAPI application.
    """
    global _config_loader
    _config_loader = config_loader
    
    app = FastAPI(
        title="mem0-server",
        description="Standalone MCP server for mem0 memory management",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize MCP server
    mcp_manager = MCPServerManager(config)
    
    # Include MCP router
    app.include_router(mcp_manager.router)
    
    # Include REST API router with full CRUD capabilities
    from mem0_server.api.routes import create_api_router
    
    api_router = create_api_router(
        config=config,
        config_loader=config_loader or ConfigLoader(),
        get_memory_client_func=mcp_manager.get_memory_client_safe,
    )
    app.include_router(api_router)
    
    # Root health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "service": "mem0-server"}
    
    # Redirect root to docs
    @app.get("/")
    async def root():
        return {
            "service": "mem0-server",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
            "api": "/api/v1",
            "mcp": "/mcp/{client_name}/sse/{user_id}",
        }
    
    return app


def run_server(config: Mem0ServerConfig, config_loader: ConfigLoader | None = None) -> None:
    """Run the MCP server with uvicorn.
    
    Args:
        config: The server configuration.
        config_loader: Optional config loader for saving configuration changes.
    """
    import uvicorn
    
    app = create_app(config, config_loader)
    
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        log_level=config.server.log_level,
    )
