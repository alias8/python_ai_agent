from contextlib import asynccontextmanager
from typing import Any

from anthropic import AsyncAnthropic
from fastapi import FastAPI

from .direct_api.agent import run_agent
from .schemas import ChatRequest, ChatResponse
from .shared.config import get_settings
from .shared.trace import get_trace_file_path

_clients: dict[str, AsyncAnthropic] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    # One client for the process lifetime rather than one per request — it
    # holds its own connection pool, so creating a fresh one per request
    # would throw that pooling away.
    settings = get_settings()
    _clients["anthropic"] = AsyncAnthropic(api_key=settings.anthropic_api_key)
    yield
    await _clients["anthropic"].close()
    _clients.clear()


app = FastAPI(
    title="Agent Service",
    description="FastAPI sibling of ai-agent-demo — a Claude tool-calling agent loop.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    client = _clients["anthropic"]
    answer = await run_agent(client, request.message, allowed_tools=request.allowed_tools)
    return ChatResponse(answer=answer, trace_file=str(get_trace_file_path()))
