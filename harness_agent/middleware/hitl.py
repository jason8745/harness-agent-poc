"""Human-in-the-loop middleware: pauses before high-risk tool calls for approval."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from ..tools.filesystem import HIGH_RISK_TOOLS

console = Console()


def request_approval(tool_calls: list[dict[str, Any]]) -> bool:
    """Display pending tool calls and ask the user to approve or reject.

    All tool calls are shown at once; one decision covers all of them.
    Returns True if approved, False if rejected.
    """
    console.print()
    console.print(Panel.fit(
        "[bold yellow]Agent wants to write files[/bold yellow]\n"
        "Review the operations below before approving.",
        title="[bold]Approval Required[/bold]",
        border_style="yellow",
    ))

    for i, tc in enumerate(tool_calls, 1):
        name = tc.get("name", "unknown")
        args = tc.get("args", {})

        console.print(f"\n[bold cyan]{i}. {name}[/bold cyan]")

        if "file_path" in args:
            console.print(f"   Path: [green]{args['file_path']}[/green]")

        if "content" in args:
            preview = args["content"][:500]
            if len(args["content"]) > 500:
                preview += "\n... (truncated)"
            console.print(Syntax(preview, "markdown", theme="monokai", line_numbers=False))

    console.print()
    while True:
        choice = console.input(
            "[bold]Approve all? ([green]y[/green]es / [red]n[/red]o)[/bold]: "
        ).strip().lower()
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no"):
            return False
        console.print("[dim]Please enter y or n.[/dim]")
