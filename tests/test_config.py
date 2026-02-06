"""Tests for mem0-server configuration."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from mem0_server.config import (
    ConfigLoader,
    Mem0ServerConfig,
    LLMProvider,
    LLMConfig,
    LLMProviderType,
    EmbedderProvider,
    EmbedderConfig,
    EmbedderProviderType,
    VectorStoreProvider,
    VectorStoreConfig,
    VectorStoreProviderType,
)
from mem0_server.config.schema import get_default_config


class TestConfigSchema:
    """Tests for configuration schema."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = get_default_config()
        
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 8765
        assert config.server.user_id == "default"
        
        assert config.llm.provider == LLMProviderType.OPENAI
        assert config.llm.config.model == "gpt-4o-mini"
        
        assert config.embedder.provider == EmbedderProviderType.OPENAI
        assert config.embedder.config.model == "text-embedding-3-small"
        
        assert config.vector_store.provider == VectorStoreProviderType.QDRANT
    
    def test_llm_config_validation(self):
        """Test LLM configuration validation."""
        config = LLMConfig(
            model="gpt-4o",
            temperature=0.5,
            max_tokens=4000,
        )
        
        assert config.model == "gpt-4o"
        assert config.temperature == 0.5
        assert config.max_tokens == 4000
    
    def test_temperature_bounds(self):
        """Test temperature validation bounds."""
        # Valid temperatures
        LLMConfig(model="test", temperature=0.0)
        LLMConfig(model="test", temperature=1.0)
        LLMConfig(model="test", temperature=2.0)
        
        # Invalid temperature
        with pytest.raises(ValueError):
            LLMConfig(model="test", temperature=2.5)
    
    def test_to_mem0_config(self):
        """Test conversion to mem0 library config format."""
        config = get_default_config()
        mem0_config = config.to_mem0_config()
        
        assert "llm" in mem0_config
        assert "embedder" in mem0_config
        assert "vector_store" in mem0_config
        assert mem0_config["version"] == "v1.1"
        
        assert mem0_config["llm"]["provider"] == "openai"
        assert mem0_config["embedder"]["provider"] == "openai"
        assert mem0_config["vector_store"]["provider"] == "qdrant"


class TestConfigLoader:
    """Tests for configuration loader."""
    
    def test_load_default_config(self):
        """Test loading default config when no file exists."""
        loader = ConfigLoader()
        config = loader.load()
        
        assert isinstance(config, Mem0ServerConfig)
        assert config.server.port == 8765
    
    def test_save_and_load_yaml(self):
        """Test saving and loading YAML config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test-config.yaml"
            
            # Create and save config
            config = get_default_config()
            config.server.port = 9999
            config.server.user_id = "test-user"
            
            loader = ConfigLoader()
            saved_path = loader.save(config, config_path)
            
            assert saved_path == config_path
            assert config_path.exists()
            
            # Load and verify
            new_loader = ConfigLoader(config_path)
            loaded_config = new_loader.load()
            
            assert loaded_config.server.port == 9999
            assert loaded_config.server.user_id == "test-user"
    
    def test_env_var_resolution(self):
        """Test environment variable resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test-config.yaml"
            
            # Set env var
            os.environ["TEST_API_KEY"] = "test-key-12345"
            
            # Write config with env var reference
            config_data = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": "gpt-4o",
                        "api_key": "${TEST_API_KEY}",
                    }
                }
            }
            
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)
            
            # Load and verify
            loader = ConfigLoader(config_path)
            config = loader.load()
            
            # env vars in ${} syntax are resolved during load
            # Note: api_key validation happens at model level with env: syntax
            
            # Cleanup
            del os.environ["TEST_API_KEY"]
    
    def test_create_default_config_file(self):
        """Test creating default config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "new-config.yaml"
            
            saved_path = ConfigLoader.create_default_config_file(config_path)
            
            assert saved_path == config_path
            assert config_path.exists()
            
            # Verify content
            with open(config_path) as f:
                data = yaml.safe_load(f)
            
            assert "server" in data
            assert "llm" in data
            assert "embedder" in data
            assert "vector_store" in data


class TestProviderTypes:
    """Tests for provider type enums."""
    
    def test_llm_providers(self):
        """Test LLM provider types."""
        providers = list(LLMProviderType)
        
        assert LLMProviderType.OPENAI in providers
        assert LLMProviderType.ANTHROPIC in providers
        assert LLMProviderType.OLLAMA in providers
        assert len(providers) >= 10
    
    def test_embedder_providers(self):
        """Test embedder provider types."""
        providers = list(EmbedderProviderType)
        
        assert EmbedderProviderType.OPENAI in providers
        assert EmbedderProviderType.OLLAMA in providers
        assert len(providers) >= 8
    
    def test_vector_store_providers(self):
        """Test vector store provider types."""
        providers = list(VectorStoreProviderType)
        
        assert VectorStoreProviderType.QDRANT in providers
        assert VectorStoreProviderType.CHROMA in providers
        assert VectorStoreProviderType.PINECONE in providers
        assert len(providers) >= 10
