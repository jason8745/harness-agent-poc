"""Memory middleware: loads AGENTS.md and per-repo memory into the system prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import AgentMiddleware, append_to_system

MEMORY_DIR = Path.home() / ".harness-agent"
GLOBAL_MEMORY_FILE = MEMORY_DIR / "AGENTS.md"
REPO_MEMORY_DIR = MEMORY_DIR / "memory"

_MEMORY_GUIDELINES = """<memory_guidelines>
When the user provides information useful for future analyses, update memory IMMEDIATELY
by calling write_file before doing anything else.

Update memory when:
- User states a preference about how analyses should be written
- User provides project-specific context you should remember
- You discover a recurring pattern worth noting

NEVER store API keys, passwords, or credentials.
</memory_guidelines>"""


class MemoryMiddleware(AgentMiddleware):
    """Loads global and repo-specific memory and injects them into the system prompt."""

    def __init__(self, repo_name: str | None = None):
        self.repo_name = repo_name

    def before_agent(self, state: dict[str, Any]) -> dict[str, Any] | None:
        # Only load once per session
        if "memory_loaded" in state:
            return None

        global_memory = _read_file(GLOBAL_MEMORY_FILE)
        repo_memory = _read_file(REPO_MEMORY_DIR / f"{self.repo_name}.md") if self.repo_name else ""

        return {
            "memory_loaded": True,
            "global_memory": global_memory,
            "repo_memory": repo_memory,
        }

    def inject_system(self, system: str, state: dict[str, Any]) -> str:
        parts = []

        global_mem = state.get("global_memory", "")
        repo_mem = state.get("repo_memory", "")

        memory_content = ""
        if global_mem:
            memory_content += f"{GLOBAL_MEMORY_FILE}\n{global_mem}\n"
        if repo_mem and self.repo_name:
            memory_content += f"\n{REPO_MEMORY_DIR}/{self.repo_name}.md\n{repo_mem}"

        if memory_content:
            parts.append(f"<agent_memory>\n{memory_content.strip()}\n</agent_memory>")

        parts.append(_MEMORY_GUIDELINES)

        return append_to_system(system, "\n\n".join(parts))


def _read_file(path: Path) -> str:
    """Read a file and return its content, or empty string if missing."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def ensure_memory_files(repo_name: str) -> None:
    """Create memory directory and stub files if they don't exist."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    REPO_MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    if not GLOBAL_MEMORY_FILE.exists():
        GLOBAL_MEMORY_FILE.write_text(
            "# Agent Memory\n\n## User Preferences\n\n## Notes\n", encoding="utf-8"
        )

    repo_file = REPO_MEMORY_DIR / f"{repo_name}.md"
    if not repo_file.exists():
        repo_file.write_text(
            f"# {repo_name}\n\n## Previous Analyses\n", encoding="utf-8"
        )
