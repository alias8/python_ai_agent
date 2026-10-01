"""Wraps shared/tools_core.py's business logic as LangChain tools.

There's no schema to hand-build here: @tool(..., args_schema=Model) takes
the tool's Pydantic input model directly, so the JSON Schema Claude sees is
generated from the same model that validates the input.
"""

import time
from collections.abc import Callable
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool

from ..shared.tools_core import (
    LookupOrderStatusInput,
    SearchKnowledgeBaseInput,
    SendNotificationInput,
    lookup_order_status,
    send_notification,
)
from ..shared.trace import trace_tool_call
from .context import AgentContext
from .knowledge_base import search_knowledge_base_semantic

# A per-process call sequence number for the trace's "iteration" field.
# create_agent()'s underlying LangGraph run doesn't expose "which loop
# iteration is this", so this instead numbers tool calls in the order they
# run within a single run_agent() call.
_call_sequence = 0


def reset_call_sequence() -> None:
    global _call_sequence
    _call_sequence = 0


def _traced(name: str, model: type[Any], func: Callable[[Any, ToolRuntime[AgentContext]], str]) -> Any:
    # Wraps a tools_core function with a JSONL trace_tool_call entry. This
    # doesn't retry or catch anything itself — it logs and re-raises. Retries
    # and turning failures into messages for the model are the agent's
    # middleware's job (agent.py), so each retry attempt shows up here as
    # its own trace entry.
    #
    # `runtime` is filled in by LangChain from the run's AgentContext, not by
    # the model: it's excluded from the schema Claude sees, and any argument
    # the model sends that isn't in `model`'s schema is dropped before this
    # runs. The adapter is what hands each tool the server's context.
    def adapter(runtime: ToolRuntime[AgentContext], **kwargs: Any) -> str:
        global _call_sequence
        iteration = _call_sequence
        _call_sequence += 1

        start = time.monotonic()
        try:
            output = func(model(**kwargs), runtime)
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
    description=(
        "Look up the current status of one of the customer's own orders by its order ID. "
        "Only the signed-in customer's orders can be found."
    ),
)(
    _traced(
        "lookup_order_status",
        LookupOrderStatusInput,
        lambda input, runtime: lookup_order_status(
            input, customer_id=runtime.context.customer_id, store_id=runtime.context.store_id
        ),
    )
)

search_knowledge_base_tool: BaseTool = tool(
    "search_knowledge_base",
    args_schema=SearchKnowledgeBaseInput,
    description=(
        "Search the support knowledge base of the store the customer is shopping with, for help "
        "articles related to a query. Policies differ between stores; this only ever returns the "
        "current store's."
    ),
)(
    _traced(
        "search_knowledge_base",
        SearchKnowledgeBaseInput,
        lambda input, runtime: search_knowledge_base_semantic(input, store_id=runtime.context.store_id),
    )
)

send_notification_tool: BaseTool = tool(
    "send_notification",
    args_schema=SendNotificationInput,
    description="Send a notification message to the customer, e.g. to tell them about their order.",
)(
    _traced(
        "send_notification",
        SendNotificationInput,
        # The tool call's ID is the idempotency key: the retry middleware
        # re-runs the same tool call, so every retry carries the same key.
        lambda input, runtime: send_notification(
            input, customer_id=runtime.context.customer_id, idempotency_key=str(runtime.tool_call_id)
        ),
    )
)

LANGCHAIN_TOOLS: list[BaseTool] = [
    lookup_order_status_tool,
    search_knowledge_base_tool,
    send_notification_tool,
]
