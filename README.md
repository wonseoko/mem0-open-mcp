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

# Test configuration (recommended for initial setup)
mem0-open-mcp test

# Start the server
mem0-open-mcp serve

# With options
mem0-open-mcp serve --port 8765 --user-id alice
```

The `test` command verifies your configuration without starting the server:
- Checks Vector Store, LLM, and Embedder connections
- Performs actual memory add/search operations
- Cleans up test data automatically

### Modes

**stdio Mode (for mcp-proxy or Claude Desktop)**

Run the server in stdio mode when integrating with mcp-proxy or Claude Desktop:

```bash
mem0-open-mcp stdio
mem0-open-mcp stdio --config ./config.yaml
```

Use this mode when:
- Running via mcp-proxy
- Claude Desktop subprocess integration
- Process spawns on demand
- **Performance**: Optimized for v0.2.1+ with lightweight manager startup

**serve Mode (HTTP/SSE server)**

Run a persistent HTTP server for remote access or multiple concurrent clients:

```bash
mem0-open-mcp serve --port 8765
```

Use this mode when:
- Remote access needed
- Multiple concurrent clients
- Always-on server preferred
- Custom port configuration required

### mcp-proxy Integration

Use mcp-proxy to route MCP protocol between tools and Claude Desktop. Configure your `mcp-servers.json`:

```json
{
  "mcpServers": {
    "mem0": {
      "command": "mem0-open-mcp",
      "args": ["stdio"]
    }
  }
}
```

Or with a custom config:

```json
{
  "mcpServers": {
    "mem0": {
      "command": "mem0-open-mcp",
      "args": ["stdio", "--config", "/path/to/config.yaml"]
    }
  }
}
```

The stdio mode communicates via stdin/stdout, making it ideal for process-spawned integrations.

### Update Command

Keep mem0-open-mcp up to date with the self-update feature:

```bash
# Check for available updates
mem0-open-mcp update --check

# Force update to latest version
mem0-open-mcp update --force

# Update and exit on success
mem0-open-mcp update
```

Options:
- `--check`: Only check for available updates without installing
- `--force`: Force reinstall even if already at latest version

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

## Performance Optimizations

### stdio Mode Optimizations (v0.2.1+)

The stdio mode is optimized for performance:

- **Lightweight Manager**: Reduced startup overhead compared to HTTP server
- **On-Demand Spawning**: Process spawns only when needed for MCP requests
- **No Server Overhead**: Eliminates HTTP/SSE connection management
- **Ideal for Claude Desktop**: Minimal resource footprint when integrated via mcp-proxy

Use stdio mode for optimal performance in Claude Desktop or mcp-proxy integrations.

### Performance Tips

- Use Qdrant vector store for best performance (recommended)
- Keep embedding dimensions consistent (768 or 1536)
- For large memory operations, increase vector store batch size in configuration
- Monitor Ollama performance with local models (llama3.2 recommended for speed)

## Graph Store (Experimental)

Graph store enables knowledge graph capabilities for relationship extraction between entities.

### Configuration

```yaml
graph_store:
  provider: "neo4j"
  config:
    url: "bolt://localhost:7687"
    username: "neo4j"
    password: "your-password"
```

### Installation

```bash
pip install mem0-open-mcp[neo4j]
# or
pip install mem0-open-mcp[kuzu]
```

### Limitations

> **⚠️ Important**: Graph store requires LLMs with proper **tool calling** support.
> 
> - **OpenAI models**: Full support (recommended for graph store)
> - **Ollama models**: Limited support - most models (llama3.2, llama3.1) do not follow tool schemas accurately, resulting in empty graph relations
> 
> If you need graph capabilities with local LLMs, consider using the `graph_store.llm` setting to specify a different LLM provider for graph operations only.

```yaml
# Example: Use OpenAI for graph, Ollama for everything else
llm:
  provider: "ollama"
  config:
    model: "llama3.2"

graph_store:
  provider: "neo4j"
  config:
    url: "bolt://localhost:7687"
    username: "neo4j"
    password: "password"
  llm:
    provider: "openai"
    config:
      model: "gpt-4o-mini"
```

## License

Apache 2.0
