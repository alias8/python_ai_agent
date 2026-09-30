"""Tool business logic and input models, kept separate from the LangChain
wrapping in langchain/tools.py.

Each tool's Pydantic input model is passed to LangChain as the tool's
args_schema directly, so the same model both validates Claude's tool input
and generates the JSON Schema Claude sees — no hand-written schema to drift
out of sync with it. search_knowledge_base's implementation lives in
langchain/knowledge_base.py; only its input model is here.
"""

import random

from pydantic import BaseModel, Field

from .exceptions import TransientToolError

# Toggle the tools' simulated flaky-service failures below. On by design —
# it's how the langchain/ vs. ai-agent-demo's TS langchain/ tool-error-handling
# difference documented in the README was actually found, rather than
# assumed from docs. Set to False when you want a deterministic run, e.g. stepping
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


def send_notification(input: SendNotificationInput) -> str:
    # A different flavor of flaky dependency than lookup_order_status: an
    # action with a side effect, rather than a read. Retrying this one is
    # only safe because the mock "notification service" is idempotent per
    # call in this demo — a real one would need a dedupe key to retry safely.
    if SIMULATE_TOOL_FAILURES and random.random() < 0.3:
        raise TransientToolError(f'notification service temporarily unavailable for user "{input.user_id}"')
    return f'Notification sent to {input.user_id}: "{input.message}"'
