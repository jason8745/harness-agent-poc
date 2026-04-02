"""LLM provider abstraction for Claude, OpenAI, and Gemini."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel

Provider = Literal["claude", "openai", "gemini"]

# Default models per provider
DEFAULT_MODELS: dict[Provider, str] = {
    "claude": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
}

# Config file location
CONFIG_PATH = Path.home() / ".harness-agent" / "config.toml"


def load_config() -> dict:
    """Load config from ~/.harness-agent/config.toml. Returns empty dict if missing."""
    if not CONFIG_PATH.exists():
        return {}
    if sys.version_info >= (3, 11):
        import tomllib
        return tomllib.loads(CONFIG_PATH.read_text())
    else:
        import tomli
        return tomli.loads(CONFIG_PATH.read_text())


def get_default_provider() -> Provider:
    """Return the default provider from config, falling back to 'claude'."""
    config = load_config()
    return config.get("llm", {}).get("provider", "claude")


def create_llm(provider: Provider | None = None) -> BaseChatModel:
    """Create and return a chat model for the given provider.

    Args:
        provider: One of 'claude', 'openai', 'gemini'. If None, reads from config.

    Returns:
        A LangChain chat model instance.
    """
    if provider is None:
        provider = get_default_provider()

    config = load_config()
    model_name = config.get("llm", {}).get("model") or DEFAULT_MODELS[provider]

    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, streaming=True)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, streaming=True)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, streaming=True)

    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from: claude, openai, gemini")


def ensure_config_exists(provider: Provider = "claude") -> None:
    """Create default config file if it doesn't exist."""
    if CONFIG_PATH.exists():
        return

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = f'[llm]\nprovider = "{provider}"\nmodel = ""\n'
    CONFIG_PATH.write_text(content)
