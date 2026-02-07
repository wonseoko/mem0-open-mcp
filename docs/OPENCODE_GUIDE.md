# OpenCode Integration Guide: mem0-open-mcp

This guide provides step-by-step instructions for integrating **mem0-open-mcp** with **OpenCode** to give your AI assistant a persistent, personalized memory layer.

## 1. Overview

**mem0-open-mcp** is an open-source Model Context Protocol (MCP) server for [Mem0](https://mem0.ai). It serves as a long-term memory layer that allows your AI agents to:
- **Remember user preferences** across sessions.
- **Track project context** and architectural decisions.
- **Learn from past interactions** to provide more relevant assistance.

### Key Benefits
- **Local LLMs**: Use Ollama or LMStudio to keep everything local.
- **Privacy**: Your memories stay on your own infrastructure.
- **Flexibility**: Works with various vector stores like Qdrant, Chroma, and more.
- **Seamless Integration**: Designed specifically to work with MCP-compatible clients like OpenCode.

---

## 2. Setup

### Prerequisites
- Python 3.10+
- A running LLM backend (Ollama is highly recommended)
- A vector store (Qdrant is recommended for performance)

### Installation
Install the package via pip:
```bash
pip install mem0-open-mcp
```

### Configuration
1. **Initialize the configuration**:
   ```bash
   mem0-open-mcp init
   ```
   This creates a default `mem0-open-mcp.yaml` in your current directory.

2. **Configure via Wizard**:
   ```bash
   mem0-open-mcp configure
   ```
   Follow the interactive prompts to set up your LLM provider, embedder, and vector store.

3. **Verify Setup**:
   ```bash
   mem0-open-mcp test
   ```
   This ensures all connections are working correctly before starting the server.

---

## 3. mcp-proxy Integration

OpenCode works best with MCP servers when routed through `mcp-proxy`. This tool manages the connection and provides a stable SSE (Server-Sent Events) endpoint.

### Install mcp-proxy
```bash
pip install mcp-proxy
```

### Configure mcp-servers.json
Create a file named `mcp-servers.json` (or add to your existing one):
```json
{
  "mcpServers": {
    "mem0": {
      "command": "mem0-open-mcp",
      "args": ["stdio", "--config", "/path/to/your/mem0-open-mcp.yaml"]
    }
  }
}
```

### Start mcp-proxy
```bash
mcp-proxy --port 8282 --named-server-config mcp-servers.json
```

---

## 4. OpenCode Configuration

Once `mcp-proxy` is running, you need to tell OpenCode where to find the Mem0 service.

1. Open your OpenCode MCP configuration file (typically located at `~/.config/opencode/mcp_config.json`).
2. Add the Mem0 server entry:
   ```json
   {
     "mcpServers": {
       "mem0": {
         "url": "http://localhost:8282/servers/mem0/sse"
       }
     }
   }
   ```
3. **Restart OpenCode** to apply the changes.

---

## 5. Available Tools

The following tools will be available to OpenCode:

| Tool | Description | Example Usage |
|------|-------------|---------------|
| `add_memories` | Stores information for future recall. | "Remember that I prefer using TypeScript for all my web projects." |
| `search_memory` | Finds relevant memories based on a query. | "What are my coding preferences?" |
| `list_memories` | Displays all stored memories for the current user. | "Show me all the memories you have about me." |
| `get_memory` | Retrieves a specific memory using its ID. | (Internal use for precise lookups) |
| `delete_memories` | Removes specific memories by ID. | "Forget the memory where I said I like Python." |
| `delete_all_memories`| Clears all memories for the current user. | "Reset all your memories of me." |

---

## 6. Usage Examples

### Scenario A: Remembering User Preferences
**User:** "I'm starting a new project. Remember that I always use Tailwind CSS for styling and Vitest for testing."
**OpenCode:** (Calls `add_memories`) "Got it! I've saved those preferences for your future projects."

### Scenario B: Tracking Project Context
**User:** "Remind me what we decided about the database schema for the user-auth module."
**OpenCode:** (Calls `search_memory` with query "user-auth database schema") "Based on our previous discussion, you decided to use PostgreSQL with Prisma as the ORM..."

---

## 7. Configuration Tips

- **Recommended Models**: 
  - For **Speed**: Use Ollama's `llama3.2`.
  - For **Quality**: Use Ollama's `qwen2.5:7b` or `llama3.1:8b`.
- **Vector Store**: Qdrant is the recommended choice for local deployments due to its efficiency and ease of use.
- **Graph Store (Optional)**: If you enable the experimental Graph Store, it's best to use an OpenAI-compatible API (like `gpt-4o-mini`) for relationship extraction, as small local models may struggle with the complex tool-calling required.

---

## 8. Troubleshooting

- **Server Connection Failed**: Ensure `mcp-proxy` is running and the port (e.g., 8282) matches your OpenCode config.
- **Memories Not Saving**: Run `mem0-open-mcp test` to verify your LLM and Vector Store connections.
- **Slow Responses**: Check if your local LLM (Ollama) is running on a GPU. Using smaller models like `llama3.2` can also help.

---

## 9. Advanced Usage

- **Remote Deployment**: You can run `mcp-proxy` and `mem0-open-mcp` on a home server and access it from OpenCode by replacing `localhost` with your server's IP.
- **Custom Metadata**: Developers can extend the server to include custom metadata (like `project_id`) for better memory organization.
- **Performance Monitoring**: Version v0.2.2+ includes enhanced logging. Check your console output to monitor request latency and memory extraction success.
