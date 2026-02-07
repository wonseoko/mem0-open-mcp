# Troubleshooting Guide: mem0-open-mcp

This guide helps you diagnose and resolve common issues with **mem0-open-mcp**.

## 1. Common Issues

### 1.1 Connection Issues

*   **mcp-proxy not connecting**: 
    *   **Diagnosis**: Ensure `mcp-proxy` is actually running and listening on the expected port (default 8282).
    *   **Solution**: Check for port conflicts using `lsof -i :8282` (macOS/Linux). Restart `mcp-proxy` if necessary.
*   **OpenCode can't find mem0 server**: 
    *   **Diagnosis**: Verify the URL in your `mcp_config.json` matches the proxy's endpoint.
    *   **Solution**: Typically `http://localhost:8282/servers/mem0/sse`. Check `mcp_config.json` for JSON syntax errors (missing commas, etc.).
*   **stdio process spawn failures**: 
    *   **Diagnosis**: Check if `mem0-open-mcp` is in your system `PATH`.
    *   **Solution**: Run `which mem0-open-mcp` to verify installation. Ensure you have the necessary permissions to execute the command. If using a specific config file, verify the path provided in `mcp-servers.json` is absolute.

### 1.2 Service Connectivity

*   **Ollama connection refused**: 
    *   **Diagnosis**: Ollama might not be running or is listening on a different port.
    *   **Solution**: Run `curl http://localhost:11434/api/tags` to check status. Verify `base_url` in your `mem0-open-mcp.yaml`.
*   **Qdrant connection timeout**: 
    *   **Diagnosis**: Qdrant service is down or unreachable.
    *   **Solution**: Run `curl http://localhost:6333/collections` to verify Qdrant is active. Check `host` and `port` settings in your configuration.
*   **Neo4j authentication error**: 
    *   **Diagnosis**: Incorrect credentials or protocol mismatch.
    *   **Solution**: Verify your `username` and `password` in the `graph_store` section. Ensure you're using the `bolt://` protocol for local connections.

### 1.3 Memory Operations

*   **Memories not saving**: 
    *   **Diagnosis**: Issue with LLM or Embedder configuration.
    *   **Solution**: Run `mem0-open-mcp test`. This command performs a full cycle of memory operations and will pinpoint where it fails.
*   **Search returns no results**: 
    *   **Diagnosis**: Vector store indexing issue or embedding dimension mismatch.
    *   **Solution**: Ensure `embedding_model_dims` in `vector_store` matches your embedder's output (e.g., 768 for `nomic-embed-text`).
*   **Slow memory operations**: 
    *   **Diagnosis**: Large LLM model or slow vector store response.
    *   **Solution**: Try a smaller, faster model like `llama3.2`. Check if your vector store is running on adequate hardware.

### 1.4 Graph Store Issues (Experimental)

*   **Empty graph relations**: 
    *   **Diagnosis**: Small local models (like `llama3.2`) often struggle with the complex tool calling required for relationship extraction.
    *   **Solution**: Use a more capable model for graph operations. You can configure a separate OpenAI-compatible API just for the graph store while keeping everything else local.
*   **Graph LLM separate config**:
    *   **Solution**: Use the `graph_store.llm` setting in your YAML:
        ```yaml
        graph_store:
          provider: "neo4j"
          llm:
            provider: "openai"
            config:
              model: "gpt-4o-mini"
              api_key: "your-api-key"
        ```

---

## 2. Debugging Tools

### 2.1 Built-in Commands

```bash
# Verify configuration and connectivity (Highy Recommended)
mem0-open-mcp test

# Check current version
mem0-open-mcp --version

# Update to the latest version
mem0-open-mcp update
```

### 2.2 Check Service Status

```bash
# Ollama: Verify running and list pulled models
curl http://localhost:11434/api/tags

# Qdrant: List existing collections
curl http://localhost:6333/collections

# Neo4j (if using graph store): Count nodes
cypher-shell -u neo4j -p your-password "MATCH (n) RETURN count(n) LIMIT 1"

# mcp-proxy (if used): Health check
curl http://localhost:8282/health
ps aux | grep mcp-proxy
```

### 2.3 Performance Logs (v0.2.1+)

In `stdio` mode, `mem0-open-mcp` logs initialization metrics to `stderr`. A typical log entry looks like:

```
[Performance] MCPServerManagerStdio initialization: 
  - FastMCP creation=50.2ms, 
  - Tool registration=30.1ms, 
  - Total time=80.3ms, 
  - Memory usage=45.2MB (RSS)
```

**What these metrics mean:**
- **FastMCP creation**: Time spent initializing the MCP framework.
- **Tool registration**: Time spent defining the available tools.
- **Total time**: Total startup time before the server is ready to handle requests. (Normal: 80-150ms).
- **Memory usage**: Resident Set Size (RSS) of the process. (Normal: ~45MB baseline).

---

## 3. Log Interpretation

### 3.1 Common Error Messages

- `"Failed to initialize mem0 async client"`: Usually indicates that the LLM or Embedder service is unreachable. Check your `base_url` settings.
- `"Memory system is currently unavailable"`: A transient error during startup or a connection failure. Check service logs (Ollama/Qdrant).
- `"Error during cleanup"`: A non-critical warning that occurred during resource release (v0.2.2+). Usually safe to ignore if functionality is unaffected.

### 3.2 Log Levels

- `error`: Best for `stdio` mode to reduce noise in client integrations.
- `info`: Shows initialization metrics and significant lifecycle events.
- `debug`: Detailed logs including full request/response payloads (use only for deep troubleshooting).

### 3.3 Where to Find Logs

- **stdio mode**: Logs are written to `stderr`. `mcp-proxy` typically captures and stores these.
- **serve mode**: Logs are written to `stdout/stderr` of the terminal running the command.
- **mcp-proxy logs**: Depends on your deployment method (check systemd journals, PM2 logs, or Docker container logs).

---

## 4. Performance Issues

### 4.1 Slow Initialization (stdio mode)

- **Expected**: 80-150ms (v0.2.1+).
- **If > 300ms**:
    - Check for high disk I/O.
    - Check network latency if services (Ollama/Qdrant) are on a different machine.
    - Identify the specific bottleneck (FastMCP creation vs. Tool registration) in the performance logs.

### 4.2 Slow Memory Operations

- **LLM selection**: Large models (e.g., `llama3.1:70b`) are significantly slower. Use `llama3.2` for the fastest memory extraction.
- **Vector search**: If search is slow, check Qdrant indexing status. Consider sharding if you have millions of memories.
- **Embeddings**: `nomic-embed-text` is generally fast and recommended.

### 4.3 High Memory Usage

- **Baseline**: `stdio` mode should use ~45MB (v0.2.1).
- **If > 100MB**: 
    - Check for potential memory leaks.
    - Version v0.2.2+ includes improved resource cleanup via the lifespan pattern to mitigate this.
- **serve mode**: Generally uses more memory due to the FastAPI overhead and keeping connections open.

---

## 5. Configuration Issues

### 5.1 YAML Syntax Errors

- **Common Mistakes**:
    - Incorrect indentation.
    - Missing colons after keys.
    - Using tabs instead of spaces.
- **Tip**: Use the `mem0-open-mcp configure` wizard to generate a valid configuration file automatically.

### 5.2 Environment Variable Issues

- The configuration supports `${ENV_VAR}` syntax for sensitive data like API keys.
- Ensure these variables are exported in the environment where `mem0-open-mcp` is running.

### 5.3 Model Not Found

- **Ollama**: Verify the model exists locally with `ollama list`.
- **Pulling Models**: If a model is missing, run `ollama pull llama3.2`.
- **Naming**: Ensure you use the exact name (e.g., `llama3.2` vs. `llama3.2:latest`).

---

## 6. Remote Deployment Issues

### 6.1 Network Configuration

- **Ollama Remote Access**: By default, Ollama only listens on `localhost`. Set `OLLAMA_HOST=0.0.0.0` to allow remote connections.
- **Firewalls**: Ensure the following ports are open if accessing services over a network:
    - `11434`: Ollama
    - `6333`: Qdrant
    - `7687`: Neo4j
    - `8282`: mcp-proxy

### 6.2 SSH Tunnel Setup (Optional)

For secure remote access without exposing ports publicly, use an SSH tunnel:
```bash
ssh -L 11434:localhost:11434 -L 6333:localhost:6333 user@your-remote-server
```

---

## 7. Getting Help

- **GitHub Issues**: [wonseoko/mem0-open-mcp/issues](https://github.com/wonseoko/mem0-open-mcp/issues)
- **Check list**: Before opening an issue, please:
    1. Check for existing similar issues.
    2. Include your version (`mem0-open-mcp --version`).
    3. Provide your (sanitized) configuration file.
    4. Attach relevant error logs.
