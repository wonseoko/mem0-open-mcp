# mem0-open-mcp

Open-source MCP server for [mem0](https://mem0.ai) — **local LLMs, self-hosted, Docker-free**.

Created because the official `mem0-mcp` configuration wasn't working properly for my setup.

## Features

- **Local LLMs**: Ollama (recommended), LMStudio*, or any OpenAI-compatible API
- **Self-hosted**: Your data stays on your infrastructure
- **Docker-free**: Simple `pip install` + CLI
- **Flexible**: YAML config with environment variable support
- **Multiple Vector Stores**: Qdrant, Chroma, Pinecone, and more

> *LMStudio requires JSON mode compatible models

## Quick Start

### Installation

```bash
pip install mem0-open-mcp
```

Or install from source:

```bash
git clone https://github.com/wonseoko/mem0-open-mcp.git
cd mem0-open-mcp
pip install -e .
```

### Usage

```bash
# Create default config
mem0-open-mcp init

# Interactive configuration wizard
mem0-open-mcp configure

# Start the server
mem0-open-mcp serve

# With options
mem0-open-mcp serve --port 8765 --user-id alice
```

## Configuration

Create `mem0-open-mcp.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8765
  user_id: "default"

llm:
  provider: "ollama"
  config:
    model: "llama3.2"
    base_url: "http://localhost:11434"

embedder:
  provider: "ollama"
  config:
    model: "nomic-embed-text"
    base_url: "http://localhost:11434"
    embedding_dims: 768

vector_store:
  provider: "qdrant"
  config:
    collection_name: "mem0_memories"
    host: "localhost"
    port: 6333
    embedding_model_dims: 768
```

### With LMStudio

> **⚠️ Note**: LMStudio requires a model that supports `response_format: json_object`. 
> mem0 uses structured JSON output for memory extraction. If you get `response_format` errors,
> use Ollama instead or select a model with JSON mode support in LMStudio.

```yaml
llm:
  provider: "openai"
  config:
    model: "your-model-name"
    base_url: "http://localhost:1234/v1"

embedder:
  provider: "openai"
  config:
    model: "your-embedding-model"
    base_url: "http://localhost:1234/v1"
```

## MCP Integration

Connect your MCP client to:

```
http://localhost:8765/mcp/<client-name>/sse/<user-id>
```

### Claude Desktop

```json
{
  "mcpServers": {
    "mem0": {
      "url": "http://localhost:8765/mcp/claude/sse/default"
    }
  }
}
```

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `add_memories` | Store new memories from text |
| `search_memory` | Search memories by query |
| `list_memories` | List all user memories |
| `get_memory` | Get a specific memory by ID |
| `delete_memories` | Delete memories by IDs |
| `delete_all_memories` | Delete all user memories |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/status` | GET | Server status |
| `/api/v1/config` | GET/PUT | Configuration |
| `/api/v1/memories` | GET/POST/DELETE | Memory operations |
| `/api/v1/memories/search` | POST | Search memories |

## Requirements

- Python 3.10+
- Vector store (Qdrant recommended)
- LLM server (Ollama, LMStudio, etc.)

## License

Apache 2.0
