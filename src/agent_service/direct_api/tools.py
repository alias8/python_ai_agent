"""Wraps shared/tools_core.py's business logic as Claude tool definitions.

The Pydantic input models double as the JSON Schema source
(model_json_schema()) and the runtime validator, so the schema Claude sees
and the validation that runs against its input can't drift apart the way a
hand-maintained schema and a hand-maintained type could.
"""

from typing import Any

from anthropic.types import ToolParam
from pydantic import BaseModel

from ..shared.tools_core import (
    LookupOrderStatusInput,
    SearchKnowledgeBaseInput,
    SendNotificationInput,
    lookup_order_status,
    search_knowledge_base,
    send_notification,
)

TOOL_DEFINITIONS: list[ToolParam] = [
    ToolParam(
        name="lookup_order_status",
        description="Look up the current status of a customer order by its order ID.",
        input_schema=LookupOrderStatusInput.model_json_schema(),
    ),
    ToolParam(
        name="search_knowledge_base",
        description="Search the internal support knowledge base for articles matching a query.",
        input_schema=SearchKnowledgeBaseInput.model_json_schema(),
    ),
    ToolParam(
        name="send_notification",
        description="Send a notification message to a user, e.g. to tell them about their order.",
        input_schema=SendNotificationInput.model_json_schema(),
    ),
]

_TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "lookup_order_status": LookupOrderStatusInput,
    "search_knowledge_base": SearchKnowledgeBaseInput,
    "send_notification": SendNotificationInput,
}

_TOOL_FUNCTIONS: dict[str, Any] = {
    "lookup_order_status": lookup_order_status,
    "search_knowledge_base": search_knowledge_base,
    "send_notification": send_notification,
}


def run_tool(name: str, raw_input: dict[str, Any], allowed_tools: list[str] | None = None) -> str:
    """Pure dispatch: validate Claude's chosen input against the matching
    Pydantic model, then call the matching function.

    allowed_tools is the actual permissions boundary for a given agent run —
    None means "no restriction". The agent loop also filters which tool
    schemas it shows Claude based on the same list, but that's just UX
    (Claude won't try to use what it can't see); this check is what actually
    stops a call from running, independent of whether that filtering
    happened, was bypassed, or Claude was talked into requesting a
    disallowed tool anyway (e.g. via a prompt injection).
    """
    if allowed_tools is not None and name not in allowed_tools:
        return f'Error: tool "{name}" is not permitted for this agent run'

    model = _TOOL_INPUT_MODELS.get(name)
    func = _TOOL_FUNCTIONS.get(name)
    if model is None or func is None:
        return f'Error: unknown tool "{name}"'

    try:
        validated_input = model.model_validate(raw_input)
    except Exception as exc:
        return f"Error: invalid input for tool \"{name}\": {exc}"

    result: str = func(validated_input)
    return result
