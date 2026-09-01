from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's message to the agent")
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Optional allowlist restricting which tools this run may use. Omit for no restriction.",
    )


class ChatResponse(BaseModel):
    answer: str
    trace_file: str
