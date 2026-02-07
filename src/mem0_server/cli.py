"""CLI for mem0-open-mcp using Typer."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Annotated

logger = logging.getLogger(__name__)

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from mem0_server import __version__
from mem0_server.config import (
    ConfigLoader,
    EmbedderProviderType,
    LLMProviderType,
    Mem0ServerConfig,
    VectorStoreProviderType,
)

app = typer.Typer(
    name="mem0-open-mcp",
    help="Standalone MCP server for mem0 with web configuration UI.",
    add_completion=False,
)

console = Console()

UPDATE_CHECK_CACHE = Path.home() / ".cache" / "mem0-open-mcp" / "update_check.json"
UPDATE_CHECK_INTERVAL = 86400  # 24 hours


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse version string to tuple for comparison."""
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


def _check_for_updates() -> None:
    """Check PyPI for newer version (once per day)."""
    try:
        UPDATE_CHECK_CACHE.parent.mkdir(parents=True, exist_ok=True)
        
        now = time.time()
        if UPDATE_CHECK_CACHE.exists():
            cache = json.loads(UPDATE_CHECK_CACHE.read_text())
            if now - cache.get("last_check", 0) < UPDATE_CHECK_INTERVAL:
                latest = cache.get("latest")
                if latest and _parse_version(latest) > _parse_version(__version__):
                    console.print(
                        f"[yellow]Update available: {__version__} → {latest}[/yellow]\n"
                        f"[dim]  pip install --upgrade mem0-open-mcp[/dim]\n"
                    )
                return
        
        import httpx
        resp = httpx.get("https://pypi.org/pypi/mem0-open-mcp/json", timeout=3)
        resp.raise_for_status()
        latest = resp.json()["info"]["version"]
        
        UPDATE_CHECK_CACHE.write_text(json.dumps({"last_check": now, "latest": latest}))
        
        if _parse_version(latest) > _parse_version(__version__):
            console.print(
                f"[yellow]Update available: {__version__} → {latest}[/yellow]\n"
                f"[dim]  pip install --upgrade mem0-open-mcp[/dim]\n"
            )
    except Exception:
        pass


def _model_matches(config_model: str, available_models: list[str]) -> bool:
    """Check if configured model matches any available model.
    
    Handles :latest suffix - 'llama3.2' matches 'llama3.2:latest'
    """
    config_lower = config_model.lower()
    config_with_latest = f"{config_lower}:latest" if ":" not in config_lower else config_lower
    
    for m in available_models:
        m_lower = m.lower()
        if config_lower == m_lower:
            return True
        if config_with_latest == m_lower:
            return True
        m_without_latest = m_lower.replace(":latest", "") if m_lower.endswith(":latest") else m_lower
        if config_lower == m_without_latest:
            return True
    return False


def _run_connectivity_tests(config: Mem0ServerConfig) -> bool:
    """Run connectivity tests for LLM, Embedder, and Vector Store."""
    console.print("[bold]Running connectivity tests...[/bold]\n")
    
    all_passed = True
    
    # Test Vector Store
    console.print("  [dim]Vector Store...[/dim]", end=" ")
    try:
        vs_config = config.vector_store
        if vs_config.provider.value == "qdrant":
            from qdrant_client import QdrantClient
            host = vs_config.config.host or "localhost"
            port = vs_config.config.port or 6333
            client = QdrantClient(host=host, port=port, timeout=5)
            client.get_collections()
            console.print("[green]✓ Connected[/green]")
        elif vs_config.provider.value == "chroma":
            import chromadb
            if vs_config.config.host:
                client = chromadb.HttpClient(host=vs_config.config.host, port=vs_config.config.port or 8000)
            else:
                client = chromadb.Client()
            client.heartbeat()
            console.print("[green]✓ Connected[/green]")
        else:
            console.print(f"[yellow]⚠ Skip (no test for {vs_config.provider.value})[/yellow]")
    except Exception as e:
        console.print(f"[red]✗ Failed: {e}[/red]")
        all_passed = False
    
    # Test LLM
    console.print("  [dim]LLM...[/dim]", end=" ")
    try:
        llm_config = config.llm
        if llm_config.provider.value == "ollama":
            import httpx
            base_url = llm_config.config.base_url or "http://localhost:11434"
            resp = httpx.get(f"{base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if _model_matches(llm_config.config.model, models):
                console.print(f"[green]✓ Connected ({llm_config.config.model})[/green]")
            else:
                console.print(f"[yellow]⚠ Connected but model '{llm_config.config.model}' not found[/yellow]")
                console.print(f"      [dim]Available: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}[/dim]")
        elif llm_config.provider.value in ("openai", "lmstudio"):
            import httpx
            base_url = llm_config.config.base_url or "https://api.openai.com/v1"
            headers = {}
            if llm_config.config.api_key:
                headers["Authorization"] = f"Bearer {llm_config.config.api_key}"
            resp = httpx.get(f"{base_url}/models", headers=headers, timeout=5)
            resp.raise_for_status()
            console.print(f"[green]✓ Connected ({llm_config.config.model})[/green]")
        else:
            console.print(f"[yellow]⚠ Skip (no test for {llm_config.provider.value})[/yellow]")
    except Exception as e:
        console.print(f"[red]✗ Failed: {e}[/red]")
        all_passed = False
    
    # Test Embedder
    console.print("  [dim]Embedder...[/dim]", end=" ")
    try:
        emb_config = config.embedder
        if emb_config.provider.value == "ollama":
            import httpx
            base_url = emb_config.config.base_url or "http://localhost:11434"
            resp = httpx.get(f"{base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if _model_matches(emb_config.config.model, models):
                console.print(f"[green]✓ Connected ({emb_config.config.model})[/green]")
            else:
                console.print(f"[yellow]⚠ Connected but model '{emb_config.config.model}' not found[/yellow]")
                console.print(f"      [dim]Available: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}[/dim]")
        elif emb_config.provider.value in ("openai", "lmstudio"):
            import httpx
            base_url = emb_config.config.base_url or "https://api.openai.com/v1"
            headers = {}
            if emb_config.config.api_key:
                headers["Authorization"] = f"Bearer {emb_config.config.api_key}"
            resp = httpx.get(f"{base_url}/models", headers=headers, timeout=5)
            resp.raise_for_status()
            console.print(f"[green]✓ Connected ({emb_config.config.model})[/green]")
        else:
            console.print(f"[yellow]⚠ Skip (no test for {emb_config.provider.value})[/yellow]")
    except Exception as e:
        console.print(f"[red]✗ Failed: {e}[/red]")
        all_passed = False
    
    # Test Graph Store (if configured)
    if config.graph_store:
        console.print("  [dim]Graph Store...[/dim]", end=" ")
        try:
            gs_config = config.graph_store
            if gs_config.provider.value == "neo4j":
                from langchain_neo4j import Neo4jGraph
                g = Neo4jGraph(
                    url=gs_config.config.url,
                    username=gs_config.config.username,
                    password=gs_config.config.password,
                    database=gs_config.config.database,
                )
                g.query("RETURN 1")
                console.print(f"[green]✓ Connected ({gs_config.provider.value})[/green]")
            elif gs_config.provider.value == "kuzu":
                console.print(f"[green]✓ Configured ({gs_config.provider.value})[/green]")
            else:
                console.print(f"[yellow]⚠ Skip (no test for {gs_config.provider.value})[/yellow]")
        except Exception as e:
            console.print(f"[red]✗ Failed: {e}[/red]")
            all_passed = False
    
    console.print()
    if all_passed:
        console.print("[bold green]All connectivity tests passed![/bold green]\n")
    else:
        console.print("[bold red]Some connectivity tests failed. Please check your configuration.[/bold red]\n")
    
    return all_passed


def _run_memory_tests(config: Mem0ServerConfig, custom_message: str | None = None) -> bool:
    """Run actual mem0 memory add/search tests."""
    import os
    import time
    import uuid
    console.print("[bold]Running memory tests...[/bold]\n")
    
    test_user_id = f"__test_user_{uuid.uuid4().hex[:8]}"
    default_message = "Alice is a software engineer at TechCorp. She knows Bob who is a data scientist at DataLab. They collaborate on AI projects."
    test_memory_text = custom_message if custom_message else default_message
    max_retries = 20
    retry_interval = 0.5
    
    console.print(f"  [dim]Test message:[/dim]")
    console.print(f"    [italic]\"{test_memory_text}\"[/italic]\n")
    
    try:
        console.print("  [dim]Initializing mem0 client...[/dim]", end=" ")
        os.environ["MEM0_TELEMETRY"] = "false"
        from mem0 import Memory
        mem0_config = config.to_mem0_config()
        memory = Memory.from_config(mem0_config)
        console.print("[green]✓[/green]")
        
        # 1. Add memory
        console.print("  [dim]1. Adding test memory...[/dim]", end=" ")
        add_result = memory.add(test_memory_text, user_id=test_user_id)
        memory_id = None
        if add_result and add_result.get("results") and len(add_result["results"]) > 0:
            first_result = add_result["results"][0]
            memory_id = first_result.get("id") if first_result else None
            if memory_id:
                console.print(f"[green]✓ Added (id: {memory_id[:8]}...)[/green]")
            else:
                console.print("[yellow]⚠ No memory extracted by LLM[/yellow]")
        else:
            console.print("[yellow]⚠ No memory extracted by LLM[/yellow]")
            console.print(f"    [dim]add_result: {add_result}[/dim]")
        
        # 2. List memories (with retry)
        console.print("  [dim]2. Listing memories...[/dim]", end=" ")
        stored_count = 0
        list_result = None
        for attempt in range(max_retries):
            list_result = memory.get_all(user_id=test_user_id)
            if list_result and list_result.get("results"):
                stored_count = len(list_result["results"])
                if not memory_id and stored_count > 0:
                    memory_id = list_result["results"][0].get("id")
                break
            if attempt < max_retries - 1:
                time.sleep(retry_interval)
        
        if stored_count > 0:
            console.print(f"[green]✓ Found {stored_count} memory(s)[/green]")
            # Show extracted memories
            if list_result and list_result.get("results"):
                for i, mem in enumerate(list_result["results"][:5]):
                    mem_text = mem.get("memory", "")[:50]
                    console.print(f"       [dim]{i+1}. {mem_text}...[/dim]")
        else:
            console.print("[red]✗ Memory not stored (waited 10s)[/red]")
            return False
        
        # 2b. Check graph store (if enabled)
        if hasattr(memory, 'enable_graph') and memory.enable_graph:
            console.print("  [dim]   Graph store...[/dim]", end=" ")
            if list_result and list_result.get("relations") is not None:
                relations = list_result.get("relations", [])
                relations_count = len(relations)
                console.print(f"[green]✓ Enabled ({relations_count} relation(s))[/green]")
                # Show extracted relations
                for rel in relations[:5]:
                    src = rel.get("source", "?")
                    r = rel.get("relationship", "?")
                    tgt = rel.get("target", rel.get("destination", "?"))
                    console.print(f"       [dim]• {src} --[{r}]--> {tgt}[/dim]")
            else:
                console.print("[yellow]⚠ Enabled but no relations returned[/yellow]")
        
        # 3. Search memories
        console.print("  [dim]3. Searching memories...[/dim]", end=" ")
        search_result = memory.search("test memory verification", user_id=test_user_id, limit=5)
        if search_result and search_result.get("results"):
            console.print(f"[green]✓ Found {len(search_result['results'])} result(s)[/green]")
        else:
            console.print("[yellow]⚠ No search results (indexing may be delayed)[/yellow]")
        
        # 4. Delete single memory (with retry verification)
        console.print("  [dim]4. Deleting single memory...[/dim]", end=" ")
        if memory_id:
            memory.delete(memory_id)
            # Verify deletion
            deleted = False
            for attempt in range(max_retries):
                list_result = memory.get_all(user_id=test_user_id)
                results = list_result.get("results", []) if list_result else []
                if not any(m.get("id") == memory_id for m in results):
                    deleted = True
                    break
                if attempt < max_retries - 1:
                    time.sleep(retry_interval)
            
            if deleted:
                console.print(f"[green]✓ Deleted (id: {memory_id[:8]}...)[/green]")
            else:
                console.print("[yellow]⚠ Delete called but not confirmed after 10s[/yellow]")
        else:
            console.print("[yellow]⚠ Skipped (no memory_id)[/yellow]")
        
        # 5. Cleanup remaining test data (with retry verification)
        console.print("  [dim]5. Cleaning up test data...[/dim]", end=" ")
        memory.delete_all(user_id=test_user_id)
        # Verify cleanup
        cleaned = False
        for attempt in range(max_retries):
            list_result = memory.get_all(user_id=test_user_id)
            results = list_result.get("results", []) if list_result else []
            if len(results) == 0:
                cleaned = True
                break
            if attempt < max_retries - 1:
                time.sleep(retry_interval)
        
        if cleaned:
            console.print("[green]✓ Cleaned[/green]")
        else:
            console.print(f"[yellow]⚠ Cleanup called but not confirmed after {max_retries}s[/yellow]")
        
        console.print()
        console.print("[bold green]All memory tests passed![/bold green]\n")
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Failed: {e}[/red]")
        console.print()
        console.print("[bold red]Memory test failed. Check your LLM/Embedder/VectorStore configuration.[/bold red]\n")
        return False


def _run_memory_tests_async(config: Mem0ServerConfig, custom_message: str | None = None) -> bool:
    """Run mem0 memory tests using AsyncMemory."""
    import asyncio
    return asyncio.run(_run_memory_tests_async_impl(config, custom_message))


async def _run_memory_tests_async_impl(config: Mem0ServerConfig, custom_message: str | None = None) -> bool:
    """Async implementation of memory tests."""
    import asyncio
    import os
    import uuid
    console.print("[bold]Running async memory tests...[/bold]\n")
    
    test_user_id = f"__test_user_{uuid.uuid4().hex[:8]}"
    default_message = "Alice is a software engineer at TechCorp. She knows Bob who is a data scientist at DataLab. They collaborate on AI projects."
    test_memory_text = custom_message if custom_message else default_message
    max_retries = 20
    retry_interval = 0.5
    
    console.print(f"  [dim]Test message:[/dim]")
    console.print(f"    [italic]\"{test_memory_text}\"[/italic]\n")
    
    try:
        console.print("  [dim]Initializing AsyncMemory client...[/dim]", end=" ")
        os.environ["MEM0_TELEMETRY"] = "false"
        from mem0 import AsyncMemory
        mem0_config = config.to_mem0_config()
        memory = await AsyncMemory.from_config(mem0_config)
        console.print("[green]✓[/green]")
        
        console.print("  [dim]1. Adding test memory...[/dim]", end=" ")
        add_result = await memory.add(test_memory_text, user_id=test_user_id)
        memory_id = None
        if add_result and add_result.get("results") and len(add_result["results"]) > 0:
            first_result = add_result["results"][0]
            memory_id = first_result.get("id") if first_result else None
            if memory_id:
                console.print(f"[green]✓ Added (id: {memory_id[:8]}...)[/green]")
            else:
                console.print("[yellow]⚠ No memory extracted by LLM[/yellow]")
        else:
            console.print("[yellow]⚠ No memory extracted by LLM[/yellow]")
            console.print(f"    [dim]add_result: {add_result}[/dim]")
        
        console.print("  [dim]2. Listing memories...[/dim]", end=" ")
        stored_count = 0
        list_result = None
        for attempt in range(max_retries):
            list_result = await memory.get_all(user_id=test_user_id)
            if list_result and list_result.get("results"):
                stored_count = len(list_result["results"])
                if not memory_id and stored_count > 0:
                    memory_id = list_result["results"][0].get("id")
                break
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_interval)
        
        if stored_count > 0:
            console.print(f"[green]✓ Found {stored_count} memory(s)[/green]")
            # Show extracted memories
            if list_result and list_result.get("results"):
                for i, mem in enumerate(list_result["results"][:5]):
                    mem_text = mem.get("memory", "")[:50]
                    console.print(f"       [dim]{i+1}. {mem_text}...[/dim]")
        else:
            console.print("[red]✗ Memory not stored (waited 10s)[/red]")
            return False
        
        if hasattr(memory, 'enable_graph') and memory.enable_graph:
            console.print("  [dim]   Graph store...[/dim]", end=" ")
            if list_result and list_result.get("relations") is not None:
                relations = list_result.get("relations", [])
                relations_count = len(relations)
                console.print(f"[green]✓ Enabled ({relations_count} relation(s))[/green]")
                # Show extracted relations
                for rel in relations[:5]:
                    src = rel.get("source", "?")
                    r = rel.get("relationship", "?")
                    tgt = rel.get("target", rel.get("destination", "?"))
                    console.print(f"       [dim]• {src} --[{r}]--> {tgt}[/dim]")
            else:
                console.print("[yellow]⚠ Enabled but no relations returned[/yellow]")
        
        console.print("  [dim]3. Searching memories...[/dim]", end=" ")
        search_result = await memory.search("test memory verification", user_id=test_user_id, limit=5)
        if search_result and search_result.get("results"):
            console.print(f"[green]✓ Found {len(search_result['results'])} result(s)[/green]")
        else:
            console.print("[yellow]⚠ No search results (indexing may be delayed)[/yellow]")
        
        console.print("  [dim]4. Deleting single memory...[/dim]", end=" ")
        if memory_id:
            await memory.delete(memory_id)
            deleted = False
            for attempt in range(max_retries):
                list_result = await memory.get_all(user_id=test_user_id)
                results = list_result.get("results", []) if list_result else []
                if not any(m.get("id") == memory_id for m in results):
                    deleted = True
                    break
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_interval)
            
            if deleted:
                console.print(f"[green]✓ Deleted (id: {memory_id[:8]}...)[/green]")
            else:
                console.print("[yellow]⚠ Delete called but not confirmed after 10s[/yellow]")
        else:
            console.print("[yellow]⚠ Skipped (no memory_id)[/yellow]")
        
        console.print("  [dim]5. Cleaning up test data...[/dim]", end=" ")
        await memory.delete_all(user_id=test_user_id)
        cleaned = False
        for attempt in range(max_retries):
            list_result = await memory.get_all(user_id=test_user_id)
            results = list_result.get("results", []) if list_result else []
            if len(results) == 0:
                cleaned = True
                break
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_interval)
        
        if cleaned:
            console.print("[green]✓ Cleaned[/green]")
        else:
            console.print("[yellow]⚠ Cleanup called but not confirmed after 10s[/yellow]")
        
        console.print()
        console.print("[bold green]All async memory tests passed![/bold green]\n")
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Failed: {e}[/red]")
        console.print()
        console.print("[bold red]Async memory test failed. Check your LLM/Embedder/VectorStore configuration.[/bold red]\n")
        return False


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"[bold green]mem0-open-mcp[/bold green] version [cyan]{__version__}[/cyan]")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version", "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """mem0-open-mcp: Standalone MCP server for mem0."""
    pass


@app.command()
def serve(
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c",
            help="Path to configuration file (YAML or JSON).",
            exists=True,
        ),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option("--host", "-h", help="Host to bind to."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", "-p", help="Port to listen on."),
    ] = None,
    user_id: Annotated[
        str | None,
        typer.Option("--user-id", "-u", help="Default user ID for memories."),
    ] = None,
    reload: Annotated[
        bool,
        typer.Option("--reload", "-r", help="Enable auto-reload for development."),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", "-l", help="Logging level."),
    ] = "info",
) -> None:
    """Start the MCP server.
    
    Examples:
        mem0-open-mcp serve
        mem0-open-mcp serve --port 8765 --user-id alice
        mem0-open-mcp serve --config ./my-config.yaml
    """
    # Load configuration
    loader = ConfigLoader(config_file)
    config = loader.load()
    
    # Override with CLI options
    if host:
        config.server.host = host
    if port:
        config.server.port = port
    if user_id:
        config.server.user_id = user_id
    if reload:
        config.server.reload = reload
    if log_level:
        config.server.log_level = log_level  # type: ignore
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.server.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Display startup info
    console.print(Panel.fit(
        f"[bold green]mem0-open-mcp[/bold green] v{__version__}\n"
        f"[dim]MCP server for mem0 memory management[/dim]",
        border_style="green",
    ))
    
    _check_for_updates()
    
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Host: [cyan]{config.server.host}[/cyan]")
    console.print(f"  Port: [cyan]{config.server.port}[/cyan]")
    console.print(f"  User ID: [cyan]{config.server.user_id}[/cyan]")
    console.print(f"  LLM: [cyan]{config.llm.provider.value}[/cyan] / {config.llm.config.model}")
    console.print(f"  Embedder: [cyan]{config.embedder.provider.value}[/cyan] / {config.embedder.config.model}")
    console.print(f"  Vector Store: [cyan]{config.vector_store.provider.value}[/cyan]")
    console.print()
    
    # Start the server
    try:
        from mem0_server.server import run_server
        run_server(config, loader)
    except ImportError as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        console.print("[yellow]Make sure all dependencies are installed: pip install mem0-open-mcp[/yellow]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def test(
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c",
            help="Path to configuration file (YAML or JSON).",
            exists=True,
        ),
    ] = None,
    use_async: Annotated[
        bool,
        typer.Option(
            "--async", "-a",
            help="Use AsyncMemory instead of sync Memory.",
        ),
    ] = False,
    message: Annotated[
        str | None,
        typer.Option(
            "--message", "-m",
            help="Custom test message for memory operations.",
        ),
    ] = None,
) -> None:
    """Test connectivity and memory operations without starting server.
    
    Examples:
        mem0-open-mcp test
        mem0-open-mcp test --async
        mem0-open-mcp test --config ./my-config.yaml
        mem0-open-mcp test --message "John works at Google as a senior engineer."
    """
    loader = ConfigLoader(config_file)
    config = loader.load()
    
    mode = "Async" if use_async else "Sync"
    console.print(Panel.fit(
        f"[bold green]mem0-open-mcp[/bold green] v{__version__}\n"
        f"[dim]Configuration Test ({mode})[/dim]",
        border_style="green",
    ))
    
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  LLM: [cyan]{config.llm.provider.value}[/cyan] / {config.llm.config.model}")
    console.print(f"  Embedder: [cyan]{config.embedder.provider.value}[/cyan] / {config.embedder.config.model}")
    console.print(f"  Vector Store: [cyan]{config.vector_store.provider.value}[/cyan]")
    if config.graph_store:
        console.print(f"  Graph Store: [cyan]{config.graph_store.provider.value}[/cyan]")
        if config.graph_store.llm:
            graph_llm_model = config.graph_store.llm.config.model if config.graph_store.llm.config else "N/A"
            console.print(f"    └─ Graph LLM: [cyan]{config.graph_store.llm.provider.value}[/cyan] / {graph_llm_model} [dim](separate)[/dim]")
        else:
            console.print(f"    └─ Graph LLM: [dim](using main LLM)[/dim]")
    console.print()
    
    if not _run_connectivity_tests(config):
        raise typer.Exit(1)
    
    if use_async:
        if not _run_memory_tests_async(config, message):
            raise typer.Exit(1)
    else:
        if not _run_memory_tests(config, message):
            raise typer.Exit(1)
    
    console.print("[bold green]All tests passed! Configuration is ready.[/bold green]")


@app.command()
def configure(
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c",
            help="Path to configuration file to create/edit.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o",
            help="Output path for configuration file.",
        ),
    ] = None,
) -> None:
    """Interactive configuration wizard.
    
    Creates or edits a mem0-open-mcp configuration file with guided prompts.
    
    Examples:
        mem0-open-mcp configure
        mem0-open-mcp configure --output ./my-config.yaml
        mem0-open-mcp configure --config ./existing-config.yaml
    """
    console.print(Panel.fit(
        "[bold green]mem0-open-mcp Configuration Wizard[/bold green]\n"
        "[dim]Configure your mem0 MCP server settings[/dim]",
        border_style="green",
    ))
    
    # Load existing config if provided
    loader = ConfigLoader(config_file)
    try:
        config = loader.load()
        console.print(f"\n[dim]Loaded existing config from: {loader.config_path}[/dim]\n")
    except Exception:
        config = Mem0ServerConfig()
        console.print("\n[dim]Creating new configuration...[/dim]\n")
    
    # === Server Settings ===
    console.print("[bold cyan]Server Settings[/bold cyan]")
    
    config.server.host = Prompt.ask(
        "  Host",
        default=config.server.host,
    )
    
    port_str = Prompt.ask(
        "  Port",
        default=str(config.server.port),
    )
    config.server.port = int(port_str)
    
    config.server.user_id = Prompt.ask(
        "  Default User ID",
        default=config.server.user_id,
    )
    
    # === LLM Settings ===
    console.print("\n[bold cyan]LLM Settings[/bold cyan]")
    
    llm_providers = [p.value for p in LLMProviderType]
    console.print(f"  Available providers: {', '.join(llm_providers)}")
    
    llm_provider = Prompt.ask(
        "  Provider",
        default=config.llm.provider.value,
        choices=llm_providers,
    )
    config.llm.provider = LLMProviderType(llm_provider)
    
    # Suggest default models based on provider
    default_models = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-20241022",
        "ollama": "llama3.2",
        "groq": "llama-3.1-70b-versatile",
    }
    suggested_model = default_models.get(llm_provider, config.llm.config.model)
    
    config.llm.config.model = Prompt.ask(
        "  Model",
        default=suggested_model,
    )
    
    if llm_provider != "ollama":
        api_key = Prompt.ask(
            "  API Key (or env:VAR_NAME)",
            default=config.llm.config.api_key or f"env:{llm_provider.upper()}_API_KEY",
        )
        config.llm.config.api_key = api_key
    else:
        base_url = Prompt.ask(
            "  Ollama URL",
            default=config.llm.config.base_url or "http://localhost:11434",
        )
        config.llm.config.base_url = base_url
    
    # === Embedder Settings ===
    console.print("\n[bold cyan]Embedder Settings[/bold cyan]")
    
    embedder_providers = [p.value for p in EmbedderProviderType]
    console.print(f"  Available providers: {', '.join(embedder_providers)}")
    
    embedder_provider = Prompt.ask(
        "  Provider",
        default=config.embedder.provider.value,
        choices=embedder_providers,
    )
    config.embedder.provider = EmbedderProviderType(embedder_provider)
    
    # Suggest default embedding models
    default_embedding_models = {
        "openai": "text-embedding-3-small",
        "ollama": "nomic-embed-text",
    }
    suggested_embed_model = default_embedding_models.get(embedder_provider, config.embedder.config.model)
    
    config.embedder.config.model = Prompt.ask(
        "  Model",
        default=suggested_embed_model,
    )
    
    if embedder_provider != "ollama":
        # Reuse LLM API key if same provider
        if embedder_provider == llm_provider and config.llm.config.api_key:
            default_embed_key = config.llm.config.api_key
        else:
            default_embed_key = config.embedder.config.api_key or f"env:{embedder_provider.upper()}_API_KEY"
        
        api_key = Prompt.ask(
            "  API Key (or env:VAR_NAME)",
            default=default_embed_key,
        )
        config.embedder.config.api_key = api_key
    else:
        base_url = Prompt.ask(
            "  Ollama URL",
            default=config.embedder.config.base_url or "http://localhost:11434",
        )
        config.embedder.config.base_url = base_url
    
    # === Vector Store Settings ===
    console.print("\n[bold cyan]Vector Store Settings[/bold cyan]")
    
    vs_providers = [p.value for p in VectorStoreProviderType]
    console.print(f"  Available providers: {', '.join(vs_providers)}")
    
    vs_provider = Prompt.ask(
        "  Provider",
        default=config.vector_store.provider.value,
        choices=vs_providers,
    )
    config.vector_store.provider = VectorStoreProviderType(vs_provider)
    
    config.vector_store.config.collection_name = Prompt.ask(
        "  Collection Name",
        default=config.vector_store.config.collection_name,
    )
    
    # Provider-specific settings
    if vs_provider in ("qdrant", "milvus", "chroma"):
        host = Prompt.ask(
            "  Host",
            default=config.vector_store.config.host or "localhost",
        )
        config.vector_store.config.host = host
        
        default_ports = {"qdrant": 6333, "milvus": 19530, "chroma": 8000}
        port_str = Prompt.ask(
            "  Port",
            default=str(config.vector_store.config.port or default_ports.get(vs_provider, 6333)),
        )
        config.vector_store.config.port = int(port_str)
    
    elif vs_provider == "pinecone":
        api_key = Prompt.ask(
            "  Pinecone API Key (or env:VAR_NAME)",
            default=config.vector_store.config.api_key or "env:PINECONE_API_KEY",
        )
        config.vector_store.config.api_key = api_key
    
    # === Custom Instructions ===
    console.print("\n[bold cyan]OpenMemory Settings[/bold cyan]")
    
    if Confirm.ask("  Add custom instructions for memory extraction?", default=False):
        instructions = Prompt.ask(
            "  Custom Instructions",
            default=config.openmemory.custom_instructions or "",
        )
        config.openmemory.custom_instructions = instructions if instructions else None
    
    # === Save Configuration ===
    console.print()
    
    save_path = output or config_file or Path("mem0-open-mcp.yaml")
    
    if Confirm.ask(f"Save configuration to [cyan]{save_path}[/cyan]?", default=True):
        saved_path = loader.save(config, save_path)
        console.print(f"\n[green]✓[/green] Configuration saved to [cyan]{saved_path}[/cyan]")
        console.print("\n[dim]Start the server with:[/dim]")
        console.print(f"  [bold]mem0-open-mcp serve --config {saved_path}[/bold]")
    else:
        console.print("[yellow]Configuration not saved.[/yellow]")


@app.command()
def status(
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c",
            help="Path to configuration file.",
            exists=True,
        ),
    ] = None,
) -> None:
    """Show current configuration and status.
    
    Examples:
        mem0-open-mcp status
        mem0-open-mcp status --config ./my-config.yaml
    """
    # Load configuration
    loader = ConfigLoader(config_file)
    config = loader.load()
    
    console.print(Panel.fit(
        "[bold green]mem0-open-mcp Status[/bold green]",
        border_style="green",
    ))
    
    if loader.config_path:
        console.print(f"\n[dim]Configuration file: {loader.config_path}[/dim]")
    else:
        console.print("\n[dim]No configuration file found, using defaults[/dim]")
    
    # Server config table
    table = Table(title="Server Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Host", config.server.host)
    table.add_row("Port", str(config.server.port))
    table.add_row("User ID", config.server.user_id)
    table.add_row("Log Level", config.server.log_level)
    
    console.print(table)
    
    # LLM config
    llm_table = Table(title="LLM Configuration")
    llm_table.add_column("Setting", style="cyan")
    llm_table.add_column("Value", style="green")
    
    llm_table.add_row("Provider", config.llm.provider.value)
    llm_table.add_row("Model", config.llm.config.model)
    llm_table.add_row("Temperature", str(config.llm.config.temperature))
    llm_table.add_row("Max Tokens", str(config.llm.config.max_tokens))
    llm_table.add_row("API Key", "***" if config.llm.config.api_key else "[red]Not set[/red]")
    if config.llm.config.base_url:
        llm_table.add_row("Base URL", config.llm.config.base_url)
    
    console.print(llm_table)
    
    # Embedder config
    embed_table = Table(title="Embedder Configuration")
    embed_table.add_column("Setting", style="cyan")
    embed_table.add_column("Value", style="green")
    
    embed_table.add_row("Provider", config.embedder.provider.value)
    embed_table.add_row("Model", config.embedder.config.model)
    embed_table.add_row("API Key", "***" if config.embedder.config.api_key else "[red]Not set[/red]")
    if config.embedder.config.base_url:
        embed_table.add_row("Base URL", config.embedder.config.base_url)
    
    console.print(embed_table)
    
    # Vector store config
    vs_table = Table(title="Vector Store Configuration")
    vs_table.add_column("Setting", style="cyan")
    vs_table.add_column("Value", style="green")
    
    vs_table.add_row("Provider", config.vector_store.provider.value)
    vs_table.add_row("Collection", config.vector_store.config.collection_name)
    if config.vector_store.config.host:
        vs_table.add_row("Host", config.vector_store.config.host)
    if config.vector_store.config.port:
        vs_table.add_row("Port", str(config.vector_store.config.port))
    if config.vector_store.config.url:
        vs_table.add_row("URL", config.vector_store.config.url)
    
    console.print(vs_table)
    
    # Connection test hint
    console.print("\n[dim]Run 'mem0-open-mcp serve' to start the server[/dim]")


@app.command()
def update(
    check_only: Annotated[
        bool,
        typer.Option(
            "--check", "-c",
            help="Only check for updates without installing.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f",
            help="Force reinstall even if already up to date.",
        ),
    ] = False,
) -> None:
    """Check for updates and upgrade to the latest version.
    
    Examples:
        mem0-open-mcp update
        mem0-open-mcp update --check
        mem0-open-mcp update --force
    """
    import subprocess
    import sys
    
    console.print(Panel.fit(
        "[bold green]mem0-open-mcp Update[/bold green]",
        border_style="green",
    ))
    
    console.print(f"\n  Current version: [cyan]{__version__}[/cyan]")
    
    # Check PyPI for latest version
    console.print("  Checking PyPI for latest version...", end=" ")
    try:
        import httpx
        resp = httpx.get("https://pypi.org/pypi/mem0-open-mcp/json", timeout=10)
        resp.raise_for_status()
        latest = resp.json()["info"]["version"]
        console.print(f"[green]✓[/green]")
        console.print(f"  Latest version:  [cyan]{latest}[/cyan]")
    except Exception as e:
        console.print(f"[red]✗[/red]")
        console.print(f"  [red]Failed to check PyPI: {e}[/red]")
        raise typer.Exit(1)
    
    # Compare versions
    current_parsed = _parse_version(__version__)
    latest_parsed = _parse_version(latest)
    
    console.print()
    
    if current_parsed >= latest_parsed and not force:
        console.print("[green]✓ You are already using the latest version![/green]")
        
        # Update cache
        try:
            UPDATE_CHECK_CACHE.parent.mkdir(parents=True, exist_ok=True)
            UPDATE_CHECK_CACHE.write_text(json.dumps({
                "last_check": time.time(),
                "latest": latest,
            }))
        except Exception:
            pass
        
        return
    
    if check_only:
        if current_parsed < latest_parsed:
            console.print(f"[yellow]Update available: {__version__} → {latest}[/yellow]")
            console.print("\n[dim]Run 'mem0-open-mcp update' to install the update.[/dim]")
        return
    
    # Perform upgrade
    action = "Reinstalling" if force else "Upgrading"
    console.print(f"[bold]{action} mem0-open-mcp...[/bold]\n")
    
    try:
        import shutil
        
        # Detect package manager: prefer uv if available
        uv_path = shutil.which("uv")
        
        if uv_path:
            # Use uv
            console.print("  [dim]Using uv...[/dim]")
            cmd = [uv_path, "pip", "install", "--upgrade", "mem0-open-mcp"]
            if force:
                cmd.append("--force-reinstall")
            pkg_manager = "uv"
        else:
            # Fallback to pip
            console.print("  [dim]Using pip...[/dim]")
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "mem0-open-mcp"]
            if force:
                cmd.append("--force-reinstall")
            pkg_manager = "pip"
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            console.print(f"[green]✓ Successfully upgraded to {latest}![/green]")
            console.print("\n[dim]Restart the server to use the new version.[/dim]")
            
            # Update cache
            try:
                UPDATE_CHECK_CACHE.parent.mkdir(parents=True, exist_ok=True)
                UPDATE_CHECK_CACHE.write_text(json.dumps({
                    "last_check": time.time(),
                    "latest": latest,
                }))
            except Exception:
                pass
        else:
            console.print(f"[red]✗ Upgrade failed[/red]")
            if result.stderr:
                console.print(f"[dim]{result.stderr}[/dim]")
            # Suggest alternative
            alt_cmd = "pip install --upgrade mem0-open-mcp" if pkg_manager == "uv" else "uv pip install --upgrade mem0-open-mcp"
            console.print(f"\n[dim]Try: {alt_cmd}[/dim]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]✗ Error during upgrade: {e}[/red]")
        console.print("\n[dim]Try manually:[/dim]")
        console.print("[dim]  pip install --upgrade mem0-open-mcp[/dim]")
        console.print("[dim]  uv pip install --upgrade mem0-open-mcp[/dim]")
        raise typer.Exit(1)


@app.command()
def stdio(
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c",
            help="Path to configuration file (YAML or JSON).",
            exists=True,
        ),
    ] = None,
    user_id: Annotated[
        str | None,
        typer.Option("--user-id", "-u", help="Default user ID for memories."),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", "-l", help="Logging level."),
    ] = "error",
) -> None:
    """Run MCP server in stdio mode (for mcp-proxy or Claude Desktop).
    
    This mode allows the server to be spawned as a subprocess and communicate
    via stdin/stdout using JSON-RPC messages.
    
    Examples:
        mem0-open-mcp stdio
        mem0-open-mcp stdio --config ./my-config.yaml
        mem0-open-mcp stdio --user-id alice
    """
    import asyncio
    import sys
    
    # Load configuration
    loader = ConfigLoader(config_file)
    config = loader.load()
    
    # Override with CLI options
    if user_id:
        config.server.user_id = user_id
    
    # Configure logging (stderr only, as stdout is used for MCP protocol)
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # Important: use stderr to avoid interfering with stdout
    )
    
    # Suppress update check in stdio mode
    async def run_stdio():
        """Run the MCP server in stdio mode."""
        try:
            from mcp.server.stdio import stdio_server
            from mem0_server.server import create_mcp_manager
            
            # Create MCP manager without HTTP server
            manager = create_mcp_manager(config)
            
            # Run stdio server
            async with stdio_server() as (read_stream, write_stream):
                await manager.mcp._mcp_server.run(
                    read_stream,
                    write_stream,
                    manager.mcp._mcp_server.create_initialization_options(),
                )
        except Exception as e:
            logger.error(f"Error in stdio mode: {e}")
            raise typer.Exit(1) from None
    
    # Run the async function
    try:
        asyncio.run(run_stdio())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def init(
    path: Annotated[
        Path | None,
        typer.Argument(
            help="Path to create configuration file. Defaults to ~/.config/mem0-open-mcp.yaml",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing file."),
    ] = False,
) -> None:
    """Create a default configuration file.
    
    Examples:
        mem0-open-mcp init
        mem0-open-mcp init ./config.yaml
        mem0-open-mcp init --force
    """
    default_path = Path.home() / ".config" / "mem0-open-mcp.yaml"
    target_path = path if path else default_path
    
    if target_path.exists() and not force:
        console.print(f"[red]File already exists: {target_path}[/red]")
        console.print("[dim]Use --force to overwrite[/dim]")
        raise typer.Exit(1)
    
    saved_path = ConfigLoader.create_default_config_file(target_path)
    console.print(f"[green]✓[/green] Created configuration file:")
    console.print(f"  [cyan]{saved_path.absolute()}[/cyan]")
    console.print("\n[dim]Edit the file to customize your settings, then run:[/dim]")
    console.print("  [bold]mem0-open-mcp serve[/bold]")


if __name__ == "__main__":
    app()
