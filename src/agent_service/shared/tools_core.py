"""Tool business logic and input models — framework-agnostic.

Used by both direct_api/tools.py (wrapped as Claude ToolParam + dispatch)
and langchain/tools.py (wrapped as LangChain @tool objects), so both
implementations are doing identical work underneath. Pydantic models are
shared too: direct_api generates Claude's JSON Schema straight off them via
model_json_schema(), and langchain passes them as a tool's args_schema
directly — no framework needs a hand-written schema duplicating the model.
"""

import random

from pydantic import BaseModel, Field

from .exceptions import TransientToolError

# Toggle the tools' simulated flaky-service failures below. On by design —
# it's what exercises direct_api's retry/backoff path and is how the
# langchain/ vs. ai-agent-demo's TS langchain/ tool-error-handling difference
# documented in the README was actually found, rather than assumed from
# docs. Set to False when you want a deterministic run, e.g. stepping
# through the happy path in a debugger without a ~40%/~30% chance of
# instead landing in the crash path on every call.
SIMULATE_TOOL_FAILURES = True


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
    if SIMULATE_TOOL_FAILURES and random.random() < 0.4:
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
    if SIMULATE_TOOL_FAILURES and random.random() < 0.3:
        raise TransientToolError(f'notification service temporarily unavailable for user "{input.user_id}"')
    return f'Notification sent to {input.user_id}: "{input.message}"'
