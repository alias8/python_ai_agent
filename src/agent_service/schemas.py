from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    # The customer and store come from the bearer token, never the body. An
    # unexpected field — a leftover store_id, or someone trying customer_id —
    # is a 422, not silently ignored, so the mistake is visible.
    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, description="The user's message to the agent")
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Optional allowlist restricting which tools this run may use. Omit for no restriction.",
    )


class ChatResponse(BaseModel):
    answer: str
    trace_file: str
