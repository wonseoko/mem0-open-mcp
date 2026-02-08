"""Configuration schema for mem0-server using Pydantic models."""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# LLM Providers
# =============================================================================


class LLMProviderType(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    TOGETHER = "together"
    GROQ = "groq"
    LITELLM = "litellm"
    MISTRALAI = "mistralai"
    GOOGLE_AI = "google_ai"
    AWS_BEDROCK = "aws_bedrock"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    XAI = "xai"
    LMSTUDIO = "lmstudio"


class LLMConfig(BaseModel):
    """LLM configuration settings."""

    model: str = Field(default="gpt-4o-mini", description="Model name")
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Temperature for generation"
    )
    max_tokens: int = Field(default=2000, gt=0, description="Maximum tokens to generate")
    api_key: str | None = Field(default=None, description="API key (supports env:VAR_NAME syntax)")
    base_url: str | None = Field(
        default=None, description="Base URL for the API (e.g., for Ollama)"
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_env_var(cls, v: str | None) -> str | None:
        """Resolve env:VAR_NAME syntax to actual environment variable value."""
        if v is None:
            return None
        if v.startswith("env:"):
            env_var = v[4:]
            return os.environ.get(env_var)
        return v


class LLMProvider(BaseModel):
    """LLM provider configuration."""

    provider: LLMProviderType = Field(default=LLMProviderType.OPENAI, description="LLM provider")
    config: LLMConfig = Field(
        default_factory=LLMConfig, description="Provider-specific configuration"
    )


# =============================================================================
# Embedder Providers
# =============================================================================


class EmbedderProviderType(str, Enum):
    """Supported embedder providers."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    VERTEXAI = "vertexai"
    GEMINI = "gemini"
    LMSTUDIO = "lmstudio"
    TOGETHER = "together"
    AWS_BEDROCK = "aws_bedrock"


class EmbedderConfig(BaseModel):
    """Embedder configuration settings."""

    model: str = Field(default="text-embedding-3-small", description="Embedding model name")
    api_key: str | None = Field(default=None, description="API key (supports env:VAR_NAME syntax)")
    base_url: str | None = Field(
        default=None, description="Base URL for the API (e.g., for Ollama)"
    )
    embedding_dims: int | None = Field(
        default=None, description="Embedding dimensions (auto-detected if not set)"
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_env_var(cls, v: str | None) -> str | None:
        """Resolve env:VAR_NAME syntax to actual environment variable value."""
        if v is None:
            return None
        if v.startswith("env:"):
            env_var = v[4:]
            return os.environ.get(env_var)
        return v


class EmbedderProvider(BaseModel):
    """Embedder provider configuration."""

    provider: EmbedderProviderType = Field(
        default=EmbedderProviderType.OPENAI, description="Embedder provider"
    )
    config: EmbedderConfig = Field(
        default_factory=EmbedderConfig, description="Provider-specific configuration"
    )


# =============================================================================
# Vector Store Providers
# =============================================================================


class VectorStoreProviderType(str, Enum):
    """Supported vector store providers."""

    QDRANT = "qdrant"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    MILVUS = "milvus"
    WEAVIATE = "weaviate"
    PGVECTOR = "pgvector"
    FAISS = "faiss"
    REDIS = "redis"
    AZURE_AI_SEARCH = "azure_ai_search"
    VERTEX_AI_VECTOR_SEARCH = "vertex_ai_vector_search"
    MONGODB_ATLAS = "mongodb_atlas"
    MEMORY = "memory"  # In-memory for testing


class VectorStoreConfig(BaseModel):
    """Vector store configuration settings.

    Note: Different providers require different configuration options.
    Common fields are defined here, and additional provider-specific
    config can be passed through the extra dict.
    """

    # Common settings
    collection_name: str = Field(default="mem0_memories", description="Collection/index name")

    # Qdrant/Milvus/Chroma settings
    host: str | None = Field(default=None, description="Server host (e.g., localhost)")
    port: int | None = Field(default=None, description="Server port (e.g., 6333 for Qdrant)")
    path: str | None = Field(default=None, description="Local storage path for embedded databases")

    # Cloud service settings
    api_key: str | None = Field(default=None, description="API key for cloud services")
    url: str | None = Field(default=None, description="Full URL for cloud services")

    # Embedding dimensions
    embedding_model_dims: int | None = Field(default=None, description="Embedding dimensions")

    # Extra provider-specific config
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Additional provider-specific settings"
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_env_var(cls, v: str | None) -> str | None:
        """Resolve env:VAR_NAME syntax to actual environment variable value."""
        if v is None:
            return None
        if v.startswith("env:"):
            env_var = v[4:]
            return os.environ.get(env_var)
        return v


class VectorStoreProvider(BaseModel):
    """Vector store provider configuration."""

    provider: VectorStoreProviderType = Field(
        default=VectorStoreProviderType.QDRANT, description="Vector store provider"
    )
    config: VectorStoreConfig = Field(
        default_factory=VectorStoreConfig, description="Provider-specific configuration"
    )


# =============================================================================
# Graph Store Providers
# =============================================================================


class GraphStoreProviderType(str, Enum):
    """Supported graph store providers."""

    NEO4J = "neo4j"
    MEMGRAPH = "memgraph"
    NEPTUNE = "neptune"
    KUZU = "kuzu"


class GraphStoreConfig(BaseModel):
    """Graph store configuration settings."""

    url: str | None = Field(default=None, description="Graph database URL")
    username: str | None = Field(default=None, description="Username for authentication")
    password: str | None = Field(default=None, description="Password for authentication")
    database: str | None = Field(default=None, description="Database name (Neo4j)")
    db: str | None = Field(default=None, description="Database path (Kuzu, default: :memory:)")
    endpoint: str | None = Field(default=None, description="Neptune endpoint")
    base_label: bool | None = Field(default=None, description="Use base node label __Entity__")

    @field_validator("password", mode="before")
    @classmethod
    def resolve_env_var(cls, v: str | None) -> str | None:
        """Resolve env:VAR_NAME syntax to actual environment variable value."""
        if v is None:
            return None
        if v.startswith("env:"):
            env_var = v[4:]
            return os.environ.get(env_var)
        return v


class GraphStoreLLMConfig(BaseModel):
    """LLM configuration for graph store queries."""

    provider: LLMProviderType = Field(default=LLMProviderType.OPENAI, description="LLM provider")
    config: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")


class GraphStoreProvider(BaseModel):
    """Graph store provider configuration."""

    provider: GraphStoreProviderType = Field(
        default=GraphStoreProviderType.NEO4J, description="Graph store provider"
    )
    config: GraphStoreConfig = Field(
        default_factory=GraphStoreConfig, description="Provider-specific configuration"
    )
    llm: GraphStoreLLMConfig | None = Field(
        default=None, description="LLM for graph store queries (uses main LLM if not set)"
    )
    custom_prompt: str | None = Field(
        default=None, description="Custom prompt for entity extraction"
    )
    threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Embedding similarity threshold for node matching"
    )


# =============================================================================
# Custom Prompts Settings
# =============================================================================


class CustomPromptsConfig(BaseModel):
    """Custom prompts for memory extraction and management."""

    fact_extraction: str | None = Field(
        default=None,
        description="Custom prompt for fact extraction from conversations. "
        "Should instruct the LLM to extract facts and return JSON with 'facts' key.",
    )
    update_memory: str | None = Field(
        default=None, description="Custom prompt for memory update decisions."
    )


# =============================================================================
# OpenMemory Settings
# =============================================================================


class OpenMemoryConfig(BaseModel):
    """OpenMemory-specific configuration."""

    custom_instructions: str | None = Field(
        default=None, description="Custom instructions for memory extraction and management"
    )
    custom_categories: dict[str, str] | None = Field(
        default=None, description="Custom category definitions for memory tagging"
    )


# =============================================================================
# Server Settings
# =============================================================================


class ServerConfig(BaseModel):
    """Server configuration settings."""

    host: str = Field(default="0.0.0.0", description="Server host to bind to")
    port: int = Field(default=8765, ge=1, le=65535, description="Server port")
    user_id: str = Field(default="default", description="Default user ID for memories")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="info", description="Logging level"
    )
    performance_logging: bool = Field(
        default=False,
        description="Enable per-request performance logging (tool name, duration, status)",
    )


# =============================================================================
# Main Configuration
# =============================================================================


class Mem0ServerConfig(BaseModel):
    """Complete mem0-server configuration."""

    server: ServerConfig = Field(default_factory=ServerConfig, description="Server settings")
    llm: LLMProvider = Field(default_factory=LLMProvider, description="LLM provider configuration")
    embedder: EmbedderProvider = Field(
        default_factory=EmbedderProvider, description="Embedder provider configuration"
    )
    vector_store: VectorStoreProvider = Field(
        default_factory=VectorStoreProvider, description="Vector store configuration"
    )
    openmemory: OpenMemoryConfig = Field(
        default_factory=OpenMemoryConfig, description="OpenMemory-specific settings"
    )
    graph_store: GraphStoreProvider | None = Field(
        default=None, description="Graph store configuration (optional, enables knowledge graph)"
    )
    custom_prompts: CustomPromptsConfig = Field(
        default_factory=CustomPromptsConfig,
        description="Custom prompts for fact extraction and memory updates",
    )
    vector_store: VectorStoreProvider = Field(
        default_factory=VectorStoreProvider, description="Vector store configuration"
    )
    openmemory: OpenMemoryConfig = Field(
        default_factory=OpenMemoryConfig, description="OpenMemory-specific settings"
    )
    graph_store: GraphStoreProvider | None = Field(
        default=None, description="Graph store configuration (optional, enables knowledge graph)"
    )

    def to_mem0_config(self) -> dict[str, Any]:
        """Convert to mem0 library configuration format."""
        config: dict[str, Any] = {
            "version": "v1.1",
        }

        # LLM config
        llm_config: dict[str, Any] = {
            "model": self.llm.config.model,
            "temperature": self.llm.config.temperature,
            "max_tokens": self.llm.config.max_tokens,
        }
        if self.llm.config.api_key:
            llm_config["api_key"] = self.llm.config.api_key
        if self.llm.config.base_url:
            if self.llm.provider == LLMProviderType.OLLAMA:
                llm_config["ollama_base_url"] = self.llm.config.base_url
            else:
                llm_config["openai_base_url"] = self.llm.config.base_url
                if not self.llm.config.api_key:
                    llm_config["api_key"] = "lm-studio"

        config["llm"] = {
            "provider": self.llm.provider.value,
            "config": llm_config,
        }

        # Embedder config
        embedder_config: dict[str, Any] = {
            "model": self.embedder.config.model,
        }
        if self.embedder.config.api_key:
            embedder_config["api_key"] = self.embedder.config.api_key
        if self.embedder.config.base_url:
            if self.embedder.provider == EmbedderProviderType.OLLAMA:
                embedder_config["ollama_base_url"] = self.embedder.config.base_url
            else:
                embedder_config["openai_base_url"] = self.embedder.config.base_url
                if not self.embedder.config.api_key:
                    embedder_config["api_key"] = "lm-studio"
        if self.embedder.config.embedding_dims:
            embedder_config["embedding_dims"] = self.embedder.config.embedding_dims

        config["embedder"] = {
            "provider": self.embedder.provider.value,
            "config": embedder_config,
        }

        # Vector store config
        vs_config: dict[str, Any] = {
            "collection_name": self.vector_store.config.collection_name,
        }
        if self.vector_store.config.host:
            vs_config["host"] = self.vector_store.config.host
        if self.vector_store.config.port:
            vs_config["port"] = self.vector_store.config.port
        if self.vector_store.config.path:
            vs_config["path"] = self.vector_store.config.path
        if self.vector_store.config.api_key:
            vs_config["api_key"] = self.vector_store.config.api_key
        if self.vector_store.config.url:
            vs_config["url"] = self.vector_store.config.url
        if self.vector_store.config.embedding_model_dims:
            vs_config["embedding_model_dims"] = self.vector_store.config.embedding_model_dims
        # Merge extra config
        vs_config.update(self.vector_store.config.extra)

        config["vector_store"] = {
            "provider": self.vector_store.provider.value,
            "config": vs_config,
        }

        # Custom instructions (if any) - legacy support
        if self.openmemory.custom_instructions:
            config["custom_prompt"] = self.openmemory.custom_instructions

        # Custom prompts for fact extraction and memory updates
        if self.custom_prompts.fact_extraction:
            config["custom_fact_extraction_prompt"] = self.custom_prompts.fact_extraction
        if self.custom_prompts.update_memory:
            config["custom_update_memory_prompt"] = self.custom_prompts.update_memory

        # Graph store config (if enabled)
        if self.graph_store:
            gs_config: dict[str, Any] = {}
            if self.graph_store.config.url:
                gs_config["url"] = self.graph_store.config.url
            if self.graph_store.config.username:
                gs_config["username"] = self.graph_store.config.username
            if self.graph_store.config.password:
                gs_config["password"] = self.graph_store.config.password
            if self.graph_store.config.database:
                gs_config["database"] = self.graph_store.config.database
            if self.graph_store.config.db:
                gs_config["db"] = self.graph_store.config.db
            if self.graph_store.config.endpoint:
                gs_config["endpoint"] = self.graph_store.config.endpoint
            if self.graph_store.config.base_label is not None:
                gs_config["base_label"] = self.graph_store.config.base_label

            graph_store_dict: dict[str, Any] = {
                "provider": self.graph_store.provider.value,
                "config": gs_config,
                "threshold": self.graph_store.threshold,
            }

            if self.graph_store.llm:
                llm_conf: dict[str, Any] = {
                    "model": self.graph_store.llm.config.model,
                    "temperature": self.graph_store.llm.config.temperature,
                    "max_tokens": self.graph_store.llm.config.max_tokens,
                }
                if self.graph_store.llm.config.api_key:
                    llm_conf["api_key"] = self.graph_store.llm.config.api_key
                if self.graph_store.llm.config.base_url:
                    if self.graph_store.llm.provider == LLMProviderType.OLLAMA:
                        llm_conf["ollama_base_url"] = self.graph_store.llm.config.base_url
                    else:
                        llm_conf["openai_base_url"] = self.graph_store.llm.config.base_url
                graph_store_dict["llm"] = {
                    "provider": self.graph_store.llm.provider.value,
                    "config": llm_conf,
                }

            if self.graph_store.custom_prompt:
                graph_store_dict["custom_prompt"] = self.graph_store.custom_prompt

            config["graph_store"] = graph_store_dict

        return config


def get_default_config() -> Mem0ServerConfig:
    """Get default configuration with sensible defaults."""
    return Mem0ServerConfig(
        llm=LLMProvider(
            provider=LLMProviderType.OPENAI,
            config=LLMConfig(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=2000,
                api_key="env:OPENAI_API_KEY",
            ),
        ),
        embedder=EmbedderProvider(
            provider=EmbedderProviderType.OPENAI,
            config=EmbedderConfig(
                model="text-embedding-3-small",
                api_key="env:OPENAI_API_KEY",
                embedding_dims=1536,
            ),
        ),
        vector_store=VectorStoreProvider(
            provider=VectorStoreProviderType.QDRANT,
            config=VectorStoreConfig(
                collection_name="mem0_memories",
                host="localhost",
                port=6333,
                embedding_model_dims=1536,
            ),
        ),
    )
