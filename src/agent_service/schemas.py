from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    # The customer, the store and which tools the agent may use all come from
    # the bearer token, never the body. An unexpected field — a leftover
    # store_id, someone trying customer_id, or an allowed_tools list from
    # before tools were scoped by the token — is a 422, not silently
    # ignored, so the mistake is visible.
    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, description="The user's message to the agent")


class ChatResponse(BaseModel):
    answer: str
    trace_file: str
    # This run's entries from the JSONL trace, in the order they happened:
    # one "tool_call" per attempt (so a retried call appears several times,
    # sharing a tool_call_id), then the "final_answer". A failed attempt
    # carries its exception's type, not its message.
    trace: list[dict[str, Any]]
