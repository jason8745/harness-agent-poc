"""Base class for all middleware components."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AnyMessage, SystemMessage


class AgentMiddleware:
    """Base middleware that can modify system prompt and agent state.

    Each middleware implements one or both of:
      - before_agent: run once at agent startup to initialise state
      - inject_system: append context to the system prompt before each LLM call
    """

    def before_agent(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """Called once before the first agent turn. Return state updates or None."""
        return None

    def inject_system(self, system: str, state: dict[str, Any]) -> str:
        """Called before every LLM call. Return the (possibly modified) system prompt."""
        return system


def append_to_system(system: str, text: str) -> str:
    """Append text to a system prompt string, separated by a blank line."""
    if not text:
        return system
    return f"{system}\n\n{text}" if system else text
