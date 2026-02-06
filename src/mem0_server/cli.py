"""CLI for mem0-server using Typer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

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
    name="mem0-server",
    help="Standalone MCP server for mem0 with web configuration UI.",
    add_completion=False,
)

console = Console()


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"[bold green]mem0-server[/bold green] version [cyan]{__version__}[/cyan]")
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
    """mem0-server: Standalone MCP server for mem0."""
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
        mem0-server serve
        mem0-server serve --port 8765 --user-id alice
        mem0-server serve --config ./my-config.yaml
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
        f"[bold green]mem0-server[/bold green] v{__version__}\n"
        f"[dim]MCP server for mem0 memory management[/dim]",
        border_style="green",
    ))
    
    console.print("\n[bold]Configuration:[/bold]")
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
        console.print("[yellow]Make sure all dependencies are installed: pip install mem0-server[/yellow]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None


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
    
    Creates or edits a mem0-server configuration file with guided prompts.
    
    Examples:
        mem0-server configure
        mem0-server configure --output ./my-config.yaml
        mem0-server configure --config ./existing-config.yaml
    """
    console.print(Panel.fit(
        "[bold green]mem0-server Configuration Wizard[/bold green]\n"
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
    
    save_path = output or config_file or Path("mem0-server.yaml")
    
    if Confirm.ask(f"Save configuration to [cyan]{save_path}[/cyan]?", default=True):
        saved_path = loader.save(config, save_path)
        console.print(f"\n[green]✓[/green] Configuration saved to [cyan]{saved_path}[/cyan]")
        console.print("\n[dim]Start the server with:[/dim]")
        console.print(f"  [bold]mem0-server serve --config {saved_path}[/bold]")
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
        mem0-server status
        mem0-server status --config ./my-config.yaml
    """
    # Load configuration
    loader = ConfigLoader(config_file)
    config = loader.load()
    
    console.print(Panel.fit(
        "[bold green]mem0-server Status[/bold green]",
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
    console.print("\n[dim]Run 'mem0-server serve' to start the server[/dim]")


@app.command()
def init(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to create configuration file.",
        ),
    ] = Path("mem0-server.yaml"),
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing file."),
    ] = False,
) -> None:
    """Create a default configuration file.
    
    Examples:
        mem0-server init
        mem0-server init ./config.yaml
        mem0-server init --force
    """
    if path.exists() and not force:
        console.print(f"[red]File already exists: {path}[/red]")
        console.print("[dim]Use --force to overwrite[/dim]")
        raise typer.Exit(1)
    
    saved_path = ConfigLoader.create_default_config_file(path)
    console.print(f"[green]✓[/green] Created default configuration: [cyan]{saved_path}[/cyan]")
    console.print("\n[dim]Edit the file to customize your settings, then run:[/dim]")
    console.print(f"  [bold]mem0-server serve --config {saved_path}[/bold]")


if __name__ == "__main__":
    app()
