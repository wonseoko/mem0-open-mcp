"""
mem0-server: Standalone MCP server for mem0 with web configuration UI.

This package provides a CLI tool to run mem0 as an MCP server without Docker,
with optional web UI for configuration management.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mem0-open-mcp")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
