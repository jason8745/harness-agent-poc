"""Compact middleware: auto-summarises conversation when context gets too long."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from .base import AgentMiddleware

COMPACT_THRESHOLD = 50_000   # tokens before triggering compaction
KEEP_MESSAGES = 10           # number of recent messages to keep verbatim

_SUMMARY_PROMPT = """Summarise the following conversation history concisely.
Focus on:
1. Key decisions and findings so far
2. Files already explored and what was learned
3. Current state of the analysis
4. Any pending items

Keep it brief but complete enough to continue the task.

---
{history}
"""


class CompactMiddleware(AgentMiddleware):
    """Compresses old messages into a summary when the token count exceeds the threshold."""

    def __init__(self, llm: Any):
        self.llm = llm

    def maybe_compact(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Return a (possibly compacted) message list. Runs synchronously."""
        total = count_tokens_approximately(messages)
        if total < COMPACT_THRESHOLD:
            return messages

        keep = messages[-KEEP_MESSAGES:]
        to_summarize = messages[:-KEEP_MESSAGES]

        if not to_summarize:
            return messages

        history_text = "\n".join(
            f"{m.__class__.__name__}: {_message_text(m)[:300]}"
            for m in to_summarize
        )
        prompt = _SUMMARY_PROMPT.format(history=history_text)

        response = self.llm.invoke([HumanMessage(content=prompt)])
        summary = HumanMessage(
            content=(
                f"[Context summary — {len(to_summarize)} earlier messages compacted]\n"
                f"{response.content}"
            )
        )

        print(
            f"\n[Compact] Compressed {len(to_summarize)} messages "
            f"({total:,} tokens → ~{count_tokens_approximately(keep):,} tokens kept)\n"
        )
        return [summary] + list(keep)


def _message_text(msg: AnyMessage) -> str:
    """Extract plain text from a message for summarisation."""
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        # Handle multi-part content blocks
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in msg.content
        )
    return str(msg.content)
