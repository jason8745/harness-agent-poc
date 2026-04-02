"""LangGraph agent with the full middleware stack."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from openai import BadRequestError
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .middleware.compact import CompactMiddleware
from .middleware.hitl import request_approval
from .middleware.memory import MemoryMiddleware, ensure_memory_files
from .tools.filesystem import FILESYSTEM_TOOLS, HIGH_RISK_TOOLS


# --------------------------------------------------------------------------- #
# State                                                                        #
# --------------------------------------------------------------------------- #

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Initialised once at startup
    memory_loaded: bool
    global_memory: str
    repo_memory: str
    # True once the user has approved any write_file call this session;
    # subsequent write_file calls skip the HITL prompt automatically.
    writes_approved: bool


# --------------------------------------------------------------------------- #
# Graph builder                                                                #
# --------------------------------------------------------------------------- #

def build_graph(llm: Any, repo_name: str) -> Any:
    """Build and compile the LangGraph agent graph.

    Args:
        llm: A LangChain chat model.
        repo_name: Name of the repo being analysed (used for memory + reports path).

    Returns:
        A compiled LangGraph app.
    """
    system_prompt = _load_system_prompt(repo_name)

    memory_mw = MemoryMiddleware(repo_name=repo_name)
    compact_mw = CompactMiddleware(llm=llm)
    model_with_tools = llm.bind_tools(FILESYSTEM_TOOLS)

    ensure_memory_files(repo_name)

    # ------------------------------------------------------------------ #
    # Nodes                                                                #
    # ------------------------------------------------------------------ #

    def agent_node(state: AgentState) -> dict:
        updates: dict[str, Any] = {}

        # Load memory once at the start of the session
        if not state.get("memory_loaded"):
            init = memory_mw.before_agent(state)
            if init:
                updates.update(init)
                state = {**state, **init}

        messages = compact_mw.maybe_compact(state["messages"])
        system = memory_mw.inject_system(system_prompt, state)

        try:
            response = model_with_tools.invoke(
                [SystemMessage(content=system)] + messages
            )
        except BadRequestError as e:
            # Azure content filter or other 400 errors — surface gracefully
            # instead of crashing the whole process
            error_detail = _extract_content_filter_reason(e)
            response = AIMessage(
                content=f"⚠️ Request blocked by the LLM provider: {error_detail}\n"
                        f"Try rephrasing your last message or starting a new session."
            )

        updates["messages"] = [response]
        return updates

    def tool_node_with_hitl(state: AgentState) -> dict:
        """Execute tools; ask for approval on write_file only once per session."""
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"messages": []}

        writes_approved: bool = state.get("writes_approved", False)

        risky = [tc for tc in last.tool_calls if tc["name"] in HIGH_RISK_TOOLS
                 and not writes_approved]
        safe  = [tc for tc in last.tool_calls if tc["name"] not in HIGH_RISK_TOOLS
                 or writes_approved]

        tool_messages = []
        state_updates: dict[str, Any] = {}

        # Run safe / already-approved tools immediately
        if safe:
            safe_node = ToolNode(FILESYSTEM_TOOLS)
            safe_ai = AIMessage(content="", tool_calls=safe)
            result = safe_node.invoke({"messages": state["messages"][:-1] + [safe_ai]})
            tool_messages.extend(result["messages"])

        # Handle risky tools — ask once, then remember the decision
        if risky:
            approved = request_approval(risky)

            if approved:
                state_updates["writes_approved"] = True  # skip future prompts
                risky_node = ToolNode(FILESYSTEM_TOOLS)
                risky_ai = AIMessage(content="", tool_calls=risky)
                result = risky_node.invoke({"messages": state["messages"][:-1] + [risky_ai]})
                tool_messages.extend(result["messages"])
            else:
                for tc in risky:
                    tool_messages.append(ToolMessage(
                        content="Tool call rejected by user.",
                        tool_call_id=tc["id"],
                    ))

        return {"messages": tool_messages, **state_updates}

    # ------------------------------------------------------------------ #
    # Graph assembly                                                       #
    # ------------------------------------------------------------------ #

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_with_hitl)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=MemorySaver())


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _load_system_prompt(repo_name: str) -> str:
    prompt_file = Path(__file__).parent / "prompts" / "system.md"
    text = prompt_file.read_text(encoding="utf-8")
    return text.replace("{repo_name}", repo_name)


def _extract_content_filter_reason(e: BadRequestError) -> str:
    """Pull a human-readable reason out of an Azure content filter error."""
    try:
        body = e.response.json()
        inner = body["error"].get("innererror", {})
        result = inner.get("content_filter_result", {})
        triggered = [
            f"{category}({info['severity']})"
            for category, info in result.items()
            if info.get("filtered")
        ]
        return f"content filter triggered — {', '.join(triggered)}" if triggered else str(e)
    except Exception:
        return str(e)
