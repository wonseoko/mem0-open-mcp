# mem0-open-mcp Improvements Plan

## Goal
Add performance monitoring, resource management, and comprehensive documentation to mem0-open-mcp

## Context
- Current version: v0.2.1
- Repository: /Volumes/Work/alex/_dev/mem0/mem0-open-mcp
- Deployed on m4-ai server via mcp-proxy
- stdio mode is optimized but lacks performance telemetry
- Need better resource cleanup and user documentation

## Tasks

### 1. Performance Monitoring
- [ ] Add initialization timing to MCPServerManagerStdio.__init__
- [ ] Add memory usage logging (process RSS) at startup
- [ ] Log timing breakdown: FastMCP creation, tool registration
- [ ] Add optional performance metrics to config schema
- [ ] Dependencies: None
- [ ] Parallelization: Independent
- [ ] File scope: src/mem0_server/server.py, src/mem0_server/config/schema.py

### 2. Resource Lifecycle Management
- [ ] Add __aenter__ and __aexit__ to MCPServerManagerStdio
- [ ] Implement proper AsyncMemory cleanup in __aexit__
- [ ] Add connection cleanup for vector store, graph store
- [ ] Update CLI stdio command to use async context manager
- [ ] Dependencies: Task 1 complete (for testing)
- [ ] Parallelization: Sequential after Task 1
- [ ] File scope: src/mem0_server/server.py, src/mem0_server/cli.py

### 3. OpenCode User Guide
- [ ] Create docs/OPENCODE_GUIDE.md
- [ ] Document MCP integration setup
- [ ] Add usage examples (add memory, search, etc.)
- [ ] Performance tips and best practices
- [ ] Configuration recommendations
- [ ] Dependencies: None
- [ ] Parallelization: Independent (can run parallel with Tasks 1-2)
- [ ] File scope: docs/OPENCODE_GUIDE.md (new)

### 4. Troubleshooting Guide
- [ ] Create docs/TROUBLESHOOTING.md
- [ ] Common issues and solutions
- [ ] Service connectivity debugging
- [ ] Performance troubleshooting
- [ ] Log interpretation guide
- [ ] Dependencies: Tasks 1-2 (to document new features)
- [ ] Parallelization: Can start parallel, finalize after Tasks 1-2
- [ ] File scope: docs/TROUBLESHOOTING.md (new)

### 5. Integration Testing
- [ ] Test performance logging appears in logs
- [ ] Verify resource cleanup with multiple requests
- [ ] Test stdio mode startup time measurement
- [ ] Verify documentation accuracy
- [ ] Dependencies: All tasks complete
- [ ] Parallelization: Sequential after all tasks
- [ ] File scope: Testing only, no files changed

## Execution Strategy
1. **Parallel Group 1**: Tasks 1, 3
2. **Sequential**: Task 2 (after Task 1)
3. **Parallel Group 2**: Task 4 (can start early, finalize after 1-2)
4. **Sequential**: Task 5 (after all tasks)

## Notepad Organization
- `.sisyphus/notepads/mem0-improvements/learnings.md` - Patterns and conventions
- `.sisyphus/notepads/mem0-improvements/decisions.md` - Design decisions
- `.sisyphus/notepads/mem0-improvements/issues.md` - Problems encountered
