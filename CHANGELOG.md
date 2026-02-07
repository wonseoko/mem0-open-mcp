# Changelog

All notable changes to mem0-open-mcp will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-02-07

### Added
- **Performance Monitoring**: stdio mode now logs initialization metrics
  - FastMCP creation time (~3-4ms)
  - Tool registration time (~2-3ms)
  - Total initialization time (~6-7ms)
  - Process memory usage (RSS ~75MB)
- **Resource Lifecycle Management**: Async context manager for proper cleanup
  - `MCPServerManagerStdio` now implements `__aenter__` and `__aexit__`
  - AsyncMemory client automatically cleaned up on exit
  - Prevents resource leaks in subprocess spawning scenarios
- **Documentation**: Comprehensive user guides
  - `docs/OPENCODE_GUIDE.md` - OpenCode integration guide (152 lines)
  - `docs/TROUBLESHOOTING.md` - Complete troubleshooting reference (212 lines)
- **CLI Improvements**:
  - No arguments now shows help automatically (instead of error)
  - `config` alias added for `configure` command
  - Version displayed at bottom of help screen
  - Update notification in help when new version available
  - Guides users to run `mem0-open-mcp update` for upgrades

### Changed
- `psutil>=6.0.0` added as dependency for performance monitoring
- Help screen redesigned with version and update info at bottom
- `_check_for_updates()` now returns tuple for programmatic use

### Fixed
- stdio mode exit now properly cleans up AsyncMemory connections

## [0.2.1] - 2025-XX-XX

### Added
- stdio mode optimization with lightweight manager
- Performance improvements: 38% faster startup (260ms → 100ms)
- Memory reduction: ~50% less (20MB → 10MB per spawn)

### Changed
- Created separate `MCPServerManagerStdio` class
- Removed FastAPI/SSE infrastructure from stdio mode

## [0.2.0] - 2025-XX-XX

### Added
- stdio mode support for mcp-proxy and Claude Desktop integration
- `mem0-open-mcp stdio` command for subprocess communication

### Changed
- Improved MCP protocol compatibility

## [0.1.x] - 2025-XX-XX

Initial releases with basic MCP server functionality.

[0.2.2]: https://github.com/wonseoko/mem0-open-mcp/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/wonseoko/mem0-open-mcp/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/wonseoko/mem0-open-mcp/compare/v0.1.0...v0.2.0
