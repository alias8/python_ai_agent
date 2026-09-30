from fastapi import FastAPI, HTTPException

from .langchain.agent import run_agent, select_tools
from .schemas import ChatRequest, ChatResponse
from .shared.trace import get_trace_file_path

app = FastAPI(
    title="Agent Service",
    description="FastAPI sibling of ai-agent-demo — a Claude tool-calling agent loop.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        select_tools(request.allowed_tools)
    except ValueError as exc:
        # A bad allowlist is the caller's mistake, so a 422, not a 500 from
        # deep inside the agent run.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    answer = await run_agent(request.message, allowed_tools=request.allowed_tools)
    return ChatResponse(answer=answer, trace_file=str(get_trace_file_path()))
