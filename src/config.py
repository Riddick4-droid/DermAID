"""
Central configuration loader for DermAId.
Reads configs/agent_config.yaml, expands environment variables,
and exposes a dict-like settings object.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from .logger import logger
from .exceptions import ConfigurationError

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "agent_config.yaml"

# Regex to match ${VAR_NAME} patterns
_ENV_VAR_RE = re.compile(r"\$\{([^}^{]+)\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively replace ${VAR} with environment variable values."""
    if isinstance(value, str):
        def _replace(match):
            var = match.group(1)
            return os.environ.get(var, "")
        return _ENV_VAR_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


class _Settings:
    """Singleton that holds all configuration values."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def load(self, config_path: Path = CONFIG_PATH) -> None:
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            self._data = _expand_env_vars(raw)
            logger.info("Configuration loaded from %s", config_path)
        except yaml.YAMLError as exc:
            logger.exception("Failed to parse YAML config")
            raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc

        required = ["vision", "kb", "llm", "retrieval", "agent", "parsing"]
        for section in required:
            if section not in self._data:
                raise ConfigurationError(f"Missing required config section: {section}")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._data:
            raise AttributeError(f"No config section '{name}'")
        return self._data[name]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


settings = _Settings()
try:
    settings.load()
except ConfigurationError as exc:
    logger.warning("Could not load config file: %s. Using defaults.", exc)
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
        },
        "llm": {"provider": "openai", "model_name": "gpt-3.5-turbo", "temperature": 0.3, "max_tokens": 1024},
        "retrieval": {"k": 3},
        "agent": {"session_window_size": 5, "strictness_default": 0.5, "session_dir": "sessions/"},
        "parsing": {"method": "langchain", "chunk_size": 500, "chunk_overlap": 50, "landing_ai": {}},
    })