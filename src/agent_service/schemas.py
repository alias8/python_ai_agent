from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's message to the agent")
    # Trusted because of who sends it: /chat's caller is the storefront's
    # backend, which knows which store the customer is on. It's never read
    # from the customer's message. With real auth in place, this would come
    # from the storefront's API key or the customer's session instead of the
    # body — the important part, which holds either way, is that the model
    # never supplies it.
    store_id: str = Field(..., min_length=1, description="The store (tenant) the customer is shopping with")
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Optional allowlist restricting which tools this run may use. Omit for no restriction.",
    )


class ChatResponse(BaseModel):
    answer: str
    trace_file: str
