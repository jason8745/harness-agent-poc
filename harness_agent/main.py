"""CLI entry point for the Harness Agent."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console
from rich.markdown import Markdown

from .agent import build_graph
from .llm.providers import Provider, create_llm, ensure_config_exists
from .middleware.memory import MEMORY_DIR, ensure_memory_files

console = Console()

PROVIDERS = ["claude", "openai", "azure", "gemini"]


@click.group()
def cli() -> None:
    """Harness Agent — a LangGraph-powered repo analyser."""


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option(
    "--model",
    type=click.Choice(PROVIDERS),
    default=None,
    help="LLM provider to use (overrides config default).",
)
def analyze(repo_path: str, model: str | None) -> None:
    """Analyse a local repository and generate architecture reports.

    REPO_PATH is the path to the repository root.
    """
    repo = Path(repo_path)
    repo_name = repo.name

    ensure_config_exists()
    ensure_memory_files(repo_name)

    console.print(f"\n[bold green]Harness Agent[/bold green] — analysing [cyan]{repo_name}[/cyan]")
    console.print(f"[dim]Repo path: {repo}[/dim]")
    console.print(f"[dim]Memory: {MEMORY_DIR}[/dim]\n")

    llm = create_llm(provider=model)
    app = build_graph(llm=llm, repo_name=repo_name)

    config = {"configurable": {"thread_id": f"analyze-{repo_name}"}}

    # Initial message instructs the agent to start the analysis
    initial_message = HumanMessage(
        content=(
            f"Please analyse the repository at: {repo}\n\n"
            f"The repo name is '{repo_name}'. "
            f"Write the three reports to:\n"
            f"  - reports/{repo_name}/01_architecture.md\n"
            f"  - reports/{repo_name}/02_components.md\n"
            f"  - reports/{repo_name}/03_takeaways.md\n\n"
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

    _run_agent_loop(app, state, config)


def _run_agent_loop(app: any, initial_state: dict, config: dict) -> None:
    """Stream agent responses, handle clarification questions, and loop until done."""
    current_input = initial_state

    while True:
        final_state = None

        for event in app.stream(current_input, config, stream_mode="values"):
            final_state = event
            messages = event.get("messages", [])
            if not messages:
                continue

            last = messages[-1]

            # Stream agent text responses
            if isinstance(last, AIMessage):
                if last.content and not last.tool_calls:
                    console.print()
                    console.print(Markdown(str(last.content)))
                elif last.tool_calls:
                    for tc in last.tool_calls:
                        args_preview = _format_args_preview(tc.get("args", {}))
                        console.print(
                            f"[dim]→ {tc['name']}({args_preview})[/dim]"
                        )

        if final_state is None:
            break

        # Check if the last message is the agent asking a question (no tool calls)
        messages = final_state.get("messages", [])
        last = messages[-1] if messages else None

        if isinstance(last, AIMessage) and last.content and not last.tool_calls:
            # Agent has finished or is asking for input
            if _looks_like_question(str(last.content)):
                user_reply = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
                if not user_reply:
                    break
                current_input = {"messages": [HumanMessage(content=user_reply)]}
            else:
                # Agent is done
                console.print("\n[bold green]Analysis complete.[/bold green]")
                _print_report_locations(config)
                break
        else:
            break


def _looks_like_question(text: str) -> bool:
    """Heuristic: treat agent output ending with '?' as a question needing a reply."""
    stripped = text.strip()
    return stripped.endswith("?") or stripped.endswith("?\n")


def _format_args_preview(args: dict) -> str:
    """Format tool call args into a short one-line preview."""
    parts = []
    for k, v in args.items():
        val = str(v)
        if len(val) > 60:
            val = val[:57] + "..."
        parts.append(f"{k}={val!r}")
    return ", ".join(parts)


def _print_report_locations(config: dict) -> None:
    """Print the paths of generated reports."""
    thread = config.get("configurable", {}).get("thread_id", "")
    repo_name = thread.replace("analyze-", "")
    reports_dir = Path(__file__).parent.parent / "reports" / repo_name
    if reports_dir.exists():
        console.print(f"\n[dim]Reports saved to: {reports_dir}[/dim]")
        for f in sorted(reports_dir.glob("*.md")):
            console.print(f"[dim]  {f.name}[/dim]")


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
