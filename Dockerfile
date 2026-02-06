FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install package
RUN pip install --no-cache-dir -e .

# Copy example config
COPY mem0-open-mcp.example.yaml /app/mem0-open-mcp.yaml

# Expose port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

# Run server
ENTRYPOINT ["mem0-open-mcp"]
CMD ["serve", "--config", "/app/mem0-open-mcp.yaml"]
