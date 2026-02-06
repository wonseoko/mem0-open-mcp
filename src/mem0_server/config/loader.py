"""Configuration file loader for mem0-server.

Supports loading configuration from:
- YAML files
- JSON files
- Environment variables
- Interactive prompts
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from mem0_server.config.schema import Mem0ServerConfig, get_default_config

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and save mem0-server configuration from various sources."""
    
    DEFAULT_CONFIG_PATHS = [
        Path.home() / ".config" / "mem0-open-mcp.yaml",
        Path.home() / ".config" / "mem0-open-mcp.yml",
        Path.home() / ".config" / "mem0-open-mcp.json",
        Path("mem0-open-mcp.yaml"),
        Path("mem0-open-mcp.yml"),
        Path("mem0-open-mcp.json"),
    ]
    
    def __init__(self, config_path: Path | str | None = None):
        """Initialize the config loader.
        
        Args:
            config_path: Explicit path to config file. If None, searches default locations.
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: Mem0ServerConfig | None = None
    
    @property
    def config(self) -> Mem0ServerConfig:
        """Get the current configuration, loading if necessary."""
        if self._config is None:
            self._config = self.load()
        return self._config
    
    def find_config_file(self) -> Path | None:
        """Find the first existing config file from default paths."""
        if self.config_path and self.config_path.exists():
            return self.config_path
        
        for path in self.DEFAULT_CONFIG_PATHS:
            if path.exists():
                return path
        
        return None
    
    def load(self, path: Path | str | None = None) -> Mem0ServerConfig:
        """Load configuration from file.
        
        Args:
            path: Path to config file. If None, searches default locations.
        
        Returns:
            Loaded configuration, or defaults if no config file found.
        """
        config_path = Path(path) if path else self.find_config_file()
        
        if config_path is None:
            logger.info("No config file found, using defaults")
            return get_default_config()
        
        logger.info(f"Loading config from: {config_path}")
        
        try:
            with open(config_path) as f:
                if config_path.suffix in (".yaml", ".yml"):
                    raw_config = yaml.safe_load(f)
                elif config_path.suffix == ".json":
                    raw_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            if raw_config is None:
                raw_config = {}
            
            # Resolve environment variables in the config
            resolved_config = self._resolve_env_vars(raw_config)
            
            # Parse into Pydantic model
            config = Mem0ServerConfig.model_validate(resolved_config)
            self._config = config
            self.config_path = config_path
            return config
            
        except ValidationError as e:
            logger.error(f"Config validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def save(self, config: Mem0ServerConfig, path: Path | str | None = None) -> Path:
        """Save configuration to file.
        
        Args:
            config: Configuration to save.
            path: Path to save to. If None, uses current config_path or default.
        
        Returns:
            Path where config was saved.
        """
        save_path = Path(path) if path else (self.config_path or Path("mem0-open-mcp.yaml"))
        
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict, preserving env: syntax where appropriate
        config_dict = self._config_to_saveable_dict(config)
        
        with open(save_path, "w") as f:
            if save_path.suffix in (".yaml", ".yml"):
                yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
            elif save_path.suffix == ".json":
                json.dump(config_dict, f, indent=2)
            else:
                # Default to YAML
                yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved config to: {save_path}")
        self._config = config
        self.config_path = save_path
        return save_path
    
    def _resolve_env_vars(self, obj: Any) -> Any:
        """Recursively resolve environment variables in config.
        
        Supports two syntaxes:
        - ${VAR_NAME}: Substitutes with env var value
        - env:VAR_NAME: Kept as-is for Pydantic to resolve later
        """
        if isinstance(obj, str):
            # Handle ${VAR_NAME} syntax
            def replace_env(match: re.Match[str]) -> str:
                var_name = match.group(1)
                value = os.environ.get(var_name, "")
                if not value:
                    logger.warning(f"Environment variable {var_name} is not set")
                return value
            
            return re.sub(r"\$\{([^}]+)\}", replace_env, obj)
        
        elif isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        
        elif isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        
        return obj
    
    def _config_to_saveable_dict(self, config: Mem0ServerConfig) -> dict[str, Any]:
        """Convert config to dict suitable for saving.
        
        Converts API keys back to env: syntax if they look like they came from env vars.
        """
        # Use mode="json" to convert enums to their string values for YAML serialization
        config_dict = config.model_dump(exclude_none=True, exclude_unset=False, mode="json")
        
        # Replace known API keys with env: syntax for safety
        if "llm" in config_dict and "config" in config_dict["llm"]:
            if "api_key" in config_dict["llm"]["config"]:
                # Check common env var names
                provider = config_dict["llm"].get("provider", "openai")
                env_var = self._get_api_key_env_var(provider)
                if env_var and os.environ.get(env_var):
                    config_dict["llm"]["config"]["api_key"] = f"env:{env_var}"
        
        if "embedder" in config_dict and "config" in config_dict["embedder"]:
            if "api_key" in config_dict["embedder"]["config"]:
                provider = config_dict["embedder"].get("provider", "openai")
                env_var = self._get_api_key_env_var(provider)
                if env_var and os.environ.get(env_var):
                    config_dict["embedder"]["config"]["api_key"] = f"env:{env_var}"
        
        return config_dict
    
    def _get_api_key_env_var(self, provider: str) -> str | None:
        """Get the expected environment variable name for a provider's API key."""
        provider_env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "azure_openai": "AZURE_OPENAI_API_KEY",
            "together": "TOGETHER_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistralai": "MISTRAL_API_KEY",
            "google_ai": "GOOGLE_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "xai": "XAI_API_KEY",
        }
        return provider_env_vars.get(provider.lower())
    
    @classmethod
    def create_default_config_file(cls, path: Path | str | None = None) -> Path:
        """Create a default configuration file.
        
        Args:
            path: Path to create config at. Defaults to mem0-open-mcp.yaml in current dir.
        
        Returns:
            Path where config was created.
        """
        save_path = Path(path) if path else Path("mem0-open-mcp.yaml")
        loader = cls()
        config = get_default_config()
        return loader.save(config, save_path)
