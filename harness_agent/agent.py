"""LangGraph agent with the full middleware stack."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
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
    # Set once at startup
    memory_loaded: bool
    global_memory: str
    repo_memory: str
    # Tracks auto-approved tools for this session
    auto_approved_tools: set[str]


# --------------------------------------------------------------------------- #
# Graph builder                                                                #
# --------------------------------------------------------------------------- #

def build_graph(llm: Any, repo_name: str) -> Any:
    """Build and compile the LangGraph agent graph.

    Args:
        llm: A bound LangChain chat model.
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
        # Initialise state on first call
        updates: dict[str, Any] = {}
        if not state.get("memory_loaded"):
            init = memory_mw.before_agent(state)
            if init:
                updates.update(init)
                state = {**state, **init}

        # Apply compact if needed
        messages = compact_mw.maybe_compact(state["messages"])

        # Build system prompt with memory injected
        system = memory_mw.inject_system(system_prompt, state)

        response = model_with_tools.invoke(
            [SystemMessage(content=system)] + messages
        )
        updates["messages"] = [response]
        return updates

    def tool_node_with_hitl(state: AgentState) -> dict:
        """Execute tools, pausing for human approval on high-risk ones."""
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"messages": []}

        auto_approved: set[str] = state.get("auto_approved_tools") or set()

        # Split tool calls into safe vs risky
        risky = [tc for tc in last.tool_calls if tc["name"] in HIGH_RISK_TOOLS
                 and tc["name"] not in auto_approved]
        safe = [tc for tc in last.tool_calls if tc["name"] not in HIGH_RISK_TOOLS
                or tc["name"] in auto_approved]

        tool_messages = []

        # Execute safe tools immediately
        if safe:
            safe_node = ToolNode(FILESYSTEM_TOOLS)
            # Build a synthetic AIMessage with only the safe tool calls
            safe_ai = AIMessage(content="", tool_calls=safe)
            result = safe_node.invoke({"messages": state["messages"][:-1] + [safe_ai]})
            tool_messages.extend(result["messages"])

        # Handle risky tools with HITL
        if risky:
            approved = request_approval(risky)

            if approved:
                risky_node = ToolNode(FILESYSTEM_TOOLS)
                risky_ai = AIMessage(content="", tool_calls=risky)
                result = risky_node.invoke({"messages": state["messages"][:-1] + [risky_ai]})
                tool_messages.extend(result["messages"])
            else:
                # Inject rejection messages so the agent can continue gracefully
                for tc in risky:
                    tool_messages.append(ToolMessage(
                        content="Tool call rejected by user.",
                        tool_call_id=tc["id"],
                    ))

        return {"messages": tool_messages}

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
    """Load the base system prompt and substitute the repo name."""
    prompt_file = Path(__file__).parent / "prompts" / "system.md"
    text = prompt_file.read_text(encoding="utf-8")
    return text.replace("{repo_name}", repo_name)
