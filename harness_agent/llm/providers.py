"""LLM provider abstraction for Claude, OpenAI, Azure OpenAI, and Gemini."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

# Load .env from project root on import
load_dotenv(Path(__file__).parent.parent.parent / ".env")

Provider = Literal["claude", "openai", "azure", "gemini"]

# Default models per provider (azure uses deployment name from env)
DEFAULT_MODELS: dict[str, str] = {
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


def get_default_provider() -> str:
    """Return the default provider from config, falling back to 'azure' if env is set."""
    config = load_config()
    configured = config.get("llm", {}).get("provider")
    if configured:
        return configured
    # Auto-detect azure if env vars are present
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return "azure"
    return "claude"


def create_llm(provider: str | None = None) -> BaseChatModel:
    """Create and return a chat model for the given provider.

    Args:
        provider: One of 'claude', 'openai', 'azure', 'gemini'.
                  If None, auto-detects from config or environment.

    Returns:
        A LangChain chat model instance.
    """
    if provider is None:
        provider = get_default_provider()

    config = load_config()
    model_override = config.get("llm", {}).get("model")

    if provider == "azure":
        from langchain_openai import AzureChatOpenAI

        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        api_key = os.environ["AZURE_OPENAI_API_KEY"]
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        deployment = model_override or os.environ["AZURE_OPENAI_DEPLOYMENT"]

        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version=api_version,
            api_key=api_key,
            streaming=True,
        )

    elif provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_override or DEFAULT_MODELS["claude"],
            streaming=True,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_override or DEFAULT_MODELS["openai"],
            streaming=True,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_override or DEFAULT_MODELS["gemini"],
            streaming=True,
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. Choose from: claude, openai, azure, gemini"
        )


def ensure_config_exists(provider: str = "azure") -> None:
    """Create default config file if it doesn't exist."""
    if CONFIG_PATH.exists():
        return

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = f'[llm]\nprovider = "{provider}"\nmodel = ""\n'
    CONFIG_PATH.write_text(content)
