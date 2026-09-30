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

from .db import new_session
from .exceptions import TransientToolError
from .orders import find_customer_order

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
    # No recipient field: a notification always goes to the authenticated
    # customer. When the model chose the recipient (it used to take a
    # user_id), a prompt injection could have it message anyone.
    message: str = Field(..., description="Notification message text to send to the customer")


def lookup_order_status(input: LookupOrderStatusInput, customer_id: str, store_id: str) -> str:
    """customer_id and store_id are the authenticated customer's, supplied by
    the server — they're deliberately not fields of LookupOrderStatusInput,
    the part the model fills in."""
    # Simulate the mock "order service" itself being flaky, separately from
    # an unknown order ID (handled below, and not an error — just "not
    # found"). This is the "tool is down" case a retry can plausibly fix, as
    # opposed to "tool call was wrong."
    if SIMULATE_TOOL_FAILURES and random.random() < 0.4:
        raise TransientToolError(f'order service temporarily unavailable for "{input.order_id}"')

    with new_session() as session:
        order = find_customer_order(session, input.order_id, customer_id, store_id)
    if order is None:
        # The same message whether the order doesn't exist or belongs to
        # someone else — otherwise the tool would confirm which order IDs
        # exist, and the model could be talked into saying so.
        return f'Error: no order found with ID "{input.order_id}"'
    return (
        f"order {order.id}: status={order.status}, carrier={order.carrier or 'n/a'}, "
        f"eta={order.eta or 'n/a'}"
    )


def send_notification(input: SendNotificationInput, customer_id: str) -> str:
    # A different flavor of flaky dependency than lookup_order_status: an
    # action with a side effect, rather than a read. Retrying this one is
    # only safe because the mock "notification service" is idempotent per
    # call in this demo — a real one would need a dedupe key to retry safely.
    if SIMULATE_TOOL_FAILURES and random.random() < 0.3:
        raise TransientToolError(f'notification service temporarily unavailable for "{customer_id}"')
    return f'Notification sent to {customer_id}: "{input.message}"'
