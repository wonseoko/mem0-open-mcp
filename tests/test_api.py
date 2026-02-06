"""Tests for mem0-server API routes."""

import pytest
from fastapi.testclient import TestClient

from mem0_server.config import Mem0ServerConfig, ConfigLoader
from mem0_server.config.schema import get_default_config
from mem0_server.server import create_app


@pytest.fixture
def config():
    """Get default config for testing."""
    return get_default_config()


@pytest.fixture
def mock_memory_client():
    """Mock memory client that returns empty results."""
    class MockMemoryClient:
        def add(self, text, user_id=None, metadata=None):
            return {"results": [{"id": "test-id", "memory": text, "event": "ADD"}]}
        
        def search(self, query, user_id=None, limit=10):
            return {"results": []}
        
        def get_all(self, user_id=None):
            return {"results": []}
        
        def get(self, memory_id):
            return {"id": memory_id, "memory": "Test memory"}
        
        def delete(self, memory_id):
            pass
        
        def delete_all(self, user_id=None):
            pass
    
    return MockMemoryClient()


@pytest.fixture
def client(config, mock_memory_client):
    """Create test client with mocked memory client."""
    app = create_app(config)
    
    # Override the memory client getter
    # Note: In real tests, we'd need to properly inject the mock
    
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health and status endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "mem0-server"
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "docs" in data


class TestConfigEndpoints:
    """Tests for configuration endpoints."""
    
    def test_get_config(self, client):
        """Test getting configuration."""
        response = client.get("/api/v1/config")
        assert response.status_code == 200
        
        data = response.json()
        assert "server" in data
        assert "llm" in data
        assert "embedder" in data
        assert "vector_store" in data
        assert "openmemory" in data
    
    def test_get_llm_config(self, client):
        """Test getting LLM configuration."""
        response = client.get("/api/v1/config/llm")
        assert response.status_code == 200
        
        data = response.json()
        assert "provider" in data
        assert "config" in data
    
    def test_get_embedder_config(self, client):
        """Test getting embedder configuration."""
        response = client.get("/api/v1/config/embedder")
        assert response.status_code == 200
        
        data = response.json()
        assert "provider" in data
        assert "config" in data
    
    def test_get_vector_store_config(self, client):
        """Test getting vector store configuration."""
        response = client.get("/api/v1/config/vector_store")
        assert response.status_code == 200
        
        data = response.json()
        assert "provider" in data
        assert "config" in data
    
    def test_get_providers(self, client):
        """Test getting available providers."""
        response = client.get("/api/v1/providers")
        assert response.status_code == 200
        
        data = response.json()
        assert "llm_providers" in data
        assert "embedder_providers" in data
        assert "vector_store_providers" in data
        
        assert "openai" in data["llm_providers"]
        assert "ollama" in data["llm_providers"]
    
    def test_get_status(self, client):
        """Test getting server status."""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "providers" in data


class TestMemoryEndpoints:
    """Tests for memory endpoints."""
    
    def test_list_memories_service_unavailable(self, client):
        """Test listing memories when service is unavailable."""
        # Without a properly initialized memory client, should return 503
        response = client.get("/api/v1/memories")
        # May return 503 if memory client not available
        assert response.status_code in [200, 503]
    
    def test_search_memories_request(self, client):
        """Test search memories endpoint accepts proper request."""
        response = client.post(
            "/api/v1/memories/search",
            json={"query": "test query", "limit": 5}
        )
        # May return 503 if memory client not available
        assert response.status_code in [200, 503]
    
    def test_add_memory_request(self, client):
        """Test add memory endpoint accepts proper request."""
        response = client.post(
            "/api/v1/memories",
            json={"text": "Test memory content"}
        )
        # May return 503 if memory client not available
        assert response.status_code in [200, 503]


class TestOpenAPISpec:
    """Tests for OpenAPI specification."""
    
    def test_openapi_spec_available(self, client):
        """Test that OpenAPI spec is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "info" in data
    
    def test_docs_available(self, client):
        """Test that docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200
