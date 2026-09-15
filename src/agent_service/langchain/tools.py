"""Wraps shared/tools_core.py's business logic as LangChain tools.

Unlike direct_api/tools.py, there's no schema to hand-build here either:
@tool(..., args_schema=Model) takes the same Pydantic model directly, so
Claude's JSON Schema (direct_api) and LangChain's tool schema (this file)
are generated from the exact same source, not two schemas kept in sync by
hand.
"""

import time
from typing import Any

from langchain_core.tools import BaseTool, tool

from ..shared.tools_core import (
    LookupOrderStatusInput,
    SearchKnowledgeBaseInput,
    SendNotificationInput,
    lookup_order_status,
    search_knowledge_base,
    send_notification,
)
from ..shared.trace import trace_tool_call

# A per-process call sequence number, standing in for direct_api's loop
# iteration index. create_agent()'s underlying LangGraph run doesn't expose
# "which loop iteration is this" the way our hand-rolled loop does, so this
# instead just numbers tool calls in the order they run within a single
# run_agent() call — comparable for tracing purposes, not identical.
_call_sequence = 0


def reset_call_sequence() -> None:
    global _call_sequence
    _call_sequence = 0


def _traced(name: str, model: type[Any], func: Any) -> Any:
    # Wraps a tools_core function with the same trace_tool_call entry
    # direct_api's agent loop writes, so the two are comparable side by
    # side. Unlike direct_api, this does NOT retry or catch TransientToolError
    # itself — it logs and re-raises, so LangGraph's own tool-error handling
    # is what actually determines what happens next. That's the point of
    # comparison: this only adds observability, LangChain's runtime supplies
    # whatever resilience you get without writing anything else.
    def adapter(**kwargs: Any) -> str:
        global _call_sequence
        iteration = _call_sequence
        _call_sequence += 1

        start = time.monotonic()
        try:
            output = func(model(**kwargs))
            trace_tool_call(
                iteration=iteration,
                tool=name,
                input=kwargs,
                duration_ms=int((time.monotonic() - start) * 1000),
                success=not output.startswith("Error"),
                attempts=1,
                output=output,
            )
            return str(output)
        except Exception as exc:
            trace_tool_call(
                iteration=iteration,
                tool=name,
                input=kwargs,
                duration_ms=int((time.monotonic() - start) * 1000),
                success=False,
                attempts=1,
                error=str(exc),
            )
            raise

    adapter.__name__ = name
    return adapter


lookup_order_status_tool: BaseTool = tool(
    "lookup_order_status",
    args_schema=LookupOrderStatusInput,
    description="Look up the current status of a customer order by its order ID.",
)(_traced("lookup_order_status", LookupOrderStatusInput, lookup_order_status))

search_knowledge_base_tool: BaseTool = tool(
    "search_knowledge_base",
    args_schema=SearchKnowledgeBaseInput,
    description="Search the internal support knowledge base for articles matching a query.",
)(_traced("search_knowledge_base", SearchKnowledgeBaseInput, search_knowledge_base))

send_notification_tool: BaseTool = tool(
    "send_notification",
    args_schema=SendNotificationInput,
    description="Send a notification message to a user, e.g. to tell them about their order.",
)(_traced("send_notification", SendNotificationInput, send_notification))

LANGCHAIN_TOOLS: list[BaseTool] = [
    lookup_order_status_tool,
    search_knowledge_base_tool,
    send_notification_tool,
]
