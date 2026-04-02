"""CLI entry point for the Harness Agent."""

from __future__ import annotations

from pathlib import Path

import click
from langchain_core.messages import AIMessage, HumanMessage
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import build_graph
from .llm.providers import create_llm, ensure_config_exists
from .middleware.memory import MEMORY_DIR, ensure_memory_files

console = Console()

PROVIDERS = ["claude", "openai", "azure", "gemini"]

# Directories to search for repos
SEARCH_ROOTS = [
    Path.home() / "Documents" / "GitHub",
    Path.home() / "code",
    Path.home() / "projects",
    Path.home() / "dev",
    Path.home() / "src",
    Path.home(),
]

# Analysis dimensions: key → (file, display label, description)
DIMENSIONS: dict[str, tuple[str, str, str]] = {
    "1": ("01_architecture.md", "Architecture",  "High-level design, layers, data flow, tech stack"),
    "2": ("02_components.md",   "Components",    "Key modules, responsibilities, interactions"),
    "3": ("03_takeaways.md",    "Takeaways",     "Nice patterns, clever decisions, lessons learned"),
}


@click.group()
def cli() -> None:
    """Harness Agent — a LangGraph-powered repo analyser."""


@cli.command()
@click.argument("repo_name")
@click.option(
    "--model",
    type=click.Choice(PROVIDERS),
    default=None,
    help="LLM provider (overrides config default).",
)
def analyze(repo_name: str, model: str | None) -> None:
    """Analyse a local repository by name.

    REPO_NAME is the directory name of the repository to find and analyse.
    """
    # 1. Search for the repo on the filesystem
    matches = _find_repos(repo_name)

    if not matches:
        console.print(f"[red]No repository found matching '{repo_name}'.[/red]")
        console.print("[dim]Searched in:[/dim]")
        for root in SEARCH_ROOTS:
            console.print(f"[dim]  {root}[/dim]")
        return

    # 2. HITL: confirm the correct path
    repo = _confirm_repo(matches, repo_name)
    if repo is None:
        console.print("[yellow]Cancelled.[/yellow]")
        return

    # 3. Select which dimensions to analyse
    selected = _select_dimensions()
    if not selected:
        console.print("[yellow]Cancelled.[/yellow]")
        return

    ensure_config_exists()
    ensure_memory_files(repo.name)

    console.print(
        f"\n[bold green]Harness Agent[/bold green] — analysing "
        f"[cyan]{repo.name}[/cyan]"
    )
    console.print(f"[dim]Path:   {repo}[/dim]")
    console.print(f"[dim]Memory: {MEMORY_DIR}[/dim]\n")

    llm = create_llm(provider=model)
    app = build_graph(llm=llm, repo_name=repo.name)
    config = {"configurable": {"thread_id": f"analyze-{repo.name}"}}

    # Build report file list for the selected dimensions
    report_lines = "\n".join(
        f"  - reports/{repo.name}/{file}"
        for file, _, _ in selected
    )
    dimension_labels = ", ".join(label for _, label, _ in selected)

    initial_message = HumanMessage(
        content=(
            f"Please analyse the repository at: {repo}\n\n"
            f"The repo name is '{repo.name}'. "
            f"Focus only on these dimensions: {dimension_labels}.\n\n"
            f"Write the reports to:\n{report_lines}\n\n"
            f"Start by exploring the repo structure, then read key files. "
            f"Ask me if you need clarification during the analysis."
        )
    )

    state: dict = {
        "messages": [initial_message],
        "memory_loaded": False,
        "global_memory": "",
        "repo_memory": "",
        "auto_approved_tools": set(),
    }

    _run_agent_loop(app, state, config, repo_name=repo.name)


# --------------------------------------------------------------------------- #
# Repo discovery                                                               #
# --------------------------------------------------------------------------- #

def _find_repos(name: str) -> list[Path]:
    """Search common locations for a directory matching the given repo name."""
    found: list[Path] = []
    name_lower = name.lower()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        # Search up to depth 3 to avoid being too slow
        for candidate in root.rglob("*"):
            if candidate.is_dir() and candidate.name.lower() == name_lower:
                # Skip hidden dirs and .git internals
                if any(part.startswith(".") for part in candidate.parts[-3:]):
                    continue
                found.append(candidate)
                break  # take first match per root
        if len(found) >= 5:
            break

    return found


def _confirm_repo(matches: list[Path], name: str) -> Path | None:
    """Show found paths and ask user to confirm the correct one."""
    console.print()

    if len(matches) == 1:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_row("[bold cyan]Found:[/bold cyan]", str(matches[0]))
        console.print(Panel(table, title="[bold yellow]Confirm Repository[/bold yellow]",
                             border_style="yellow"))
        console.print()
        choice = console.input("Is this correct? ([green]y[/green]es / [red]n[/red]o): ").strip().lower()
        return matches[0] if choice in ("y", "yes") else None

    # Multiple matches — let user pick
    table = Table(title=f"Found {len(matches)} matches for '{name}'", box=None)
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("Path")

    for i, path in enumerate(matches, 1):
        table.add_row(str(i), str(path))

    console.print(table)
    console.print()

    while True:
        raw = console.input(
            f"Select [bold cyan]1–{len(matches)}[/bold cyan] or [red]q[/red] to quit: "
        ).strip().lower()
        if raw == "q":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(matches):
            return matches[int(raw) - 1]
        console.print("[dim]Invalid choice, try again.[/dim]")


# --------------------------------------------------------------------------- #
# Dimension selection                                                          #
# --------------------------------------------------------------------------- #

def _select_dimensions() -> list[tuple[str, str, str]]:
    """Prompt the user to pick one or more analysis dimensions."""
    console.print()

    table = Table(title="Select analysis dimensions", box=None)
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("Dimension", style="bold")
    table.add_column("Description", style="dim")

    for key, (_, label, desc) in DIMENSIONS.items():
        table.add_row(key, label, desc)

    console.print(table)
    console.print("[dim]Enter numbers separated by spaces, or press Enter for all.[/dim]")
    console.print()

    raw = console.input("Choice [1 2 3]: ").strip()

    if not raw:
        return list(DIMENSIONS.values())

    selected = []
    for token in raw.split():
        if token in DIMENSIONS:
            selected.append(DIMENSIONS[token])

    if not selected:
        console.print("[dim]No valid selection, defaulting to all dimensions.[/dim]")
        return list(DIMENSIONS.values())

    return selected


# --------------------------------------------------------------------------- #
# Agent loop                                                                   #
# --------------------------------------------------------------------------- #

def _run_agent_loop(app: any, initial_state: dict, config: dict, repo_name: str) -> None:
    """Stream agent responses, handle clarification questions, loop until done."""
    from rich.markdown import Markdown

    current_input = initial_state

    while True:
        final_state = None

        for event in app.stream(current_input, config, stream_mode="values"):
            final_state = event
            messages = event.get("messages", [])
            if not messages:
                continue

            last = messages[-1]

            if isinstance(last, AIMessage):
                if last.content and not last.tool_calls:
                    console.print()
                    console.print(Markdown(str(last.content)))
                elif last.tool_calls:
                    for tc in last.tool_calls:
                        preview = _format_args_preview(tc.get("args", {}))
                        console.print(f"[dim]→ {tc['name']}({preview})[/dim]")

        if final_state is None:
            break

        messages = final_state.get("messages", [])
        last = messages[-1] if messages else None

        if isinstance(last, AIMessage) and last.content and not last.tool_calls:
            if _looks_like_question(str(last.content)):
                user_reply = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
                if not user_reply:
                    break
                current_input = {"messages": [HumanMessage(content=user_reply)]}
            else:
                console.print("\n[bold green]Analysis complete.[/bold green]")
                _print_report_locations(repo_name)
                break
        else:
            break


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _looks_like_question(text: str) -> bool:
    return text.strip().endswith("?")


def _format_args_preview(args: dict) -> str:
    parts = []
    for k, v in args.items():
        val = str(v)
        if len(val) > 60:
            val = val[:57] + "..."
        parts.append(f"{k}={val!r}")
    return ", ".join(parts)


def _print_report_locations(repo_name: str) -> None:
    reports_dir = Path(__file__).parent.parent / "reports" / repo_name
    if reports_dir.exists():
        console.print(f"\n[dim]Reports saved to: {reports_dir}[/dim]")
        for f in sorted(reports_dir.glob("*.md")):
            console.print(f"[dim]  {f.name}[/dim]")


# --------------------------------------------------------------------------- #
# Config command                                                               #
# --------------------------------------------------------------------------- #

@cli.command()
def config() -> None:
    """Show the current configuration and memory file paths."""
    from .llm.providers import CONFIG_PATH, get_default_provider, load_config

    ensure_config_exists()
    cfg = load_config()

    console.print(f"\n[bold]Config file:[/bold] {CONFIG_PATH}")
    console.print(f"[bold]Default provider:[/bold] {get_default_provider()}")
    console.print(f"[bold]Memory dir:[/bold] {MEMORY_DIR}")

    llm_cfg = cfg.get("llm", {})
    if llm_cfg.get("model"):
        console.print(f"[bold]Model override:[/bold] {llm_cfg['model']}")
