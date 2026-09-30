from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .langchain.agent import run_agent, select_tools
from .langchain.context import AgentContext
from .langchain.kb_corpus import store_ids
from .schemas import ChatRequest, ChatResponse
from .shared.auth import AuthError, Principal, verify_token
from .shared.trace import get_trace_file_path

app = FastAPI(
    title="Agent Service",
    description="FastAPI sibling of ai-agent-demo — a Claude tool-calling agent loop.",
)

# auto_error=False so a missing header gets the same 401 (with the
# WWW-Authenticate header) as a bad token, rather than FastAPI's 403.
_bearer = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def current_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    """Who is asking, from the storefront-issued JWT — the only source of the
    customer and store the agent runs as."""
    if credentials is None:
        raise _unauthorized("missing bearer token")
    try:
        principal = verify_token(credentials.credentials)
    except AuthError as exc:
        raise _unauthorized("invalid or expired token") from exc
    if principal.store_id not in store_ids():
        raise _unauthorized("invalid or expired token")
    return principal


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, principal: Annotated[Principal, Depends(current_principal)]
) -> ChatResponse:
    try:
        select_tools(request.allowed_tools)
    except ValueError as exc:
        # A bad allowlist is the caller's mistake, so a 422, not a 500 from
        # deep inside the agent run.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    context = AgentContext(customer_id=principal.customer_id, store_id=principal.store_id)
    answer = await run_agent(request.message, context=context, allowed_tools=request.allowed_tools)
    return ChatResponse(answer=answer, trace_file=str(get_trace_file_path()))
