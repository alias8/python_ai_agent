"""Tool business logic and Claude tool-schema definitions.

Each tool's input is a Pydantic model rather than a hand-written JSON Schema
dict — model_json_schema() generates the schema Claude sees directly from the
same type that validates the input at call time, so the two can't drift apart
the way a hand-maintained schema and a hand-maintained type hint could.
"""

import random
from typing import Any

from anthropic.types import ToolParam
from pydantic import BaseModel, Field

from .exceptions import TransientToolError


class LookupOrderStatusInput(BaseModel):
    order_id: str = Field(..., description='Order ID to look up, e.g. "ORD-1042"')


class SearchKnowledgeBaseInput(BaseModel):
    query: str = Field(..., description="Search terms to look up in the support knowledge base")


class SendNotificationInput(BaseModel):
    user_id: str = Field(..., description='ID of the user to notify, e.g. "user-88"')
    message: str = Field(..., description="Notification message text to send")


_MOCK_ORDERS: dict[str, dict[str, str]] = {
    "ord-1001": {"status": "delivered", "carrier": "UPS", "eta": "n/a"},
    "ord-1042": {"status": "in_transit", "carrier": "FedEx", "eta": "2026-09-04"},
    "ord-2003": {"status": "processing", "carrier": "n/a", "eta": "2026-09-06"},
}


def lookup_order_status(input: LookupOrderStatusInput) -> str:
    # Simulate the mock "order service" itself being flaky, separately from
    # an unknown order ID (handled below, and not an error — just "not
    # found"). This is the "tool is down" case a retry can plausibly fix, as
    # opposed to "tool call was wrong."
    if random.random() < 0.4:
        raise TransientToolError(f'order service temporarily unavailable for "{input.order_id}"')

    order = _MOCK_ORDERS.get(input.order_id.strip().lower())
    if order is None:
        return f'Error: no order found with ID "{input.order_id}"'
    return f"order {input.order_id}: {order}"


_MOCK_KB_ARTICLES: list[dict[str, str]] = [
    {
        "title": "Return policy",
        "body": "Items can be returned within 30 days of delivery for a full refund.",
    },
    {
        "title": "Shipping times",
        "body": "Standard shipping takes 3-5 business days; expedited takes 1-2.",
    },
    {
        "title": "Account recovery",
        "body": "Use the 'forgot password' link on the login page to reset your account.",
    },
]


def search_knowledge_base(input: SearchKnowledgeBaseInput) -> str:
    terms = input.query.strip().lower().split()
    matches = [
        article
        for article in _MOCK_KB_ARTICLES
        if any(term in article["title"].lower() or term in article["body"].lower() for term in terms)
    ]
    if not matches:
        return f'No knowledge base articles matched "{input.query}"'
    return "; ".join(f"{a['title']}: {a['body']}" for a in matches)


def send_notification(input: SendNotificationInput) -> str:
    # A different flavor of flaky dependency than lookup_order_status: an
    # action with a side effect, rather than a read. Retrying this one is
    # only safe because the mock "notification service" is idempotent per
    # call in this demo — a real one would need a dedupe key to retry safely.
    if random.random() < 0.3:
        raise TransientToolError(f'notification service temporarily unavailable for user "{input.user_id}"')
    return f'Notification sent to {input.user_id}: "{input.message}"'


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
