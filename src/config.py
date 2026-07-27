"""
Central configuration loader for DermAId.
Reads configs/agent_config.yaml and exposes a dict-like settings object.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from .logger import logger
from .exceptions import ConfigurationError

# Load environment variables from .env (if present)
load_dotenv()

# Determine project root (assuming this file is in src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "agent_config.yaml"


class _Settings:
    """Singleton that holds all configuration values."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def load(self, config_path: Path = CONFIG_PATH) -> None:
        """Load and validate the YAML configuration file."""
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f)
            logger.info("Configuration loaded from %s", config_path)
        except yaml.YAMLError as exc:
            logger.exception("Failed to parse YAML config")
            raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc

        # Validate required sections
        required = ["vision", "kb", "llm", "retrieval", "agent"]
        for section in required:
            if section not in self._data:
                raise ConfigurationError(f"Missing required config section: {section}")

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access, e.g., settings.vision."""
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._data:
            raise AttributeError(f"No config section '{name}'")
        return self._data[name]

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style access with optional default."""
        return self._data.get(key, default)


# Create singleton and load immediately
settings = _Settings()
try:
    settings.load()
except ConfigurationError as exc:
    logger.warning("Could not load config file: %s. Using defaults.", exc)
    # Fallback: set minimal defaults so tests can still run
    settings._data.update({
        "vision": {
            "model_name": "google/vit-base-patch16-224",
            "num_classes": 10,
            "confidence_threshold": 0.85,
        },
        "kb": {
            "collection_name": "derm_kb",
            "persist_directory": "chroma_db/",
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_separator": "\n## ",
        },
        "llm": {
            "provider": "openai",
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.3,
            "max_tokens": 1024,
        },
        "retrieval": {"k": 3},
        "agent": {
            "session_window_size": 5,
            "strictness_default": 0.5,
            "session_dir": "sessions/",
        },
    })