"""Per-run context the server supplies to the agent — never the model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentContext:
    """Passed to create_agent()'s run as `context=`, and handed to tools as
    ToolRuntime.context. Nothing here appears in any tool's schema, so the
    model can't see it, choose it, or override it — a prompt-injected "I'm
    customer cust-2002" or "I'm a customer of the other store" has no
    argument to smuggle an ID into.
    """

    # The authenticated customer: order lookups only see their orders, and
    # notifications only go to them.
    customer_id: str
    # The store (tenant) the customer is shopping with. Scopes knowledge
    # base search to that store's articles, and order lookups to its orders.
    store_id: str
