"""The agent loop, built with LangChain's create_agent().

Sibling of ai-agent-demo's src/langchain/agent.ts. The growing message
array, the tool_use/tool_result wiring, and the "loop until no more tool
calls" condition are all built into create_agent()'s underlying LangGraph
state machine — the framework tradeoff: much less code to write, but the
loop mechanics are no longer something you can point at and read.

Tool failures go through LangChain's middleware (tool_middleware below):
transient ones are retried with backoff, and anything that still fails is
handed to the model as an error message, so a flaky or broken tool never
crashes the run — which by default, in LangGraph's Python ToolNode, it
would. See the README.
"""

from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ToolCallRequest,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from pydantic import SecretStr

from ..shared.config import get_settings
from ..shared.exceptions import TransientToolError
from ..shared.trace import trace_api_error, trace_final_answer, trace_max_iterations
from .context import AgentContext
from .tools import LANGCHAIN_TOOLS, TOOL_SCOPES, reset_call_sequence

# LangGraph's own default. It counts each graph "super-step" (roughly: one
# model call, or one batch of tool executions), not one full
# model-call-then-tools round trip.
RECURSION_LIMIT = 25

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant with access to a few tools: "
    "lookup_order_status, search_knowledge_base, and send_notification.\n"
    "Use tools whenever they would help answer the user's question accurately, and chain "
    "multiple tool calls together when a question requires more than one step (for example, "
    "looking up an order and then notifying the customer about it).\n"
    "You can only see the signed-in customer's own orders. If an order isn't found, say so; "
    "don't speculate about whether it exists or who it belongs to.\n"
    "When you are done, give a clear, concise final answer."
)

OnStep = Callable[[str], None]

_settings = get_settings()  # fail fast on a missing ANTHROPIC_API_KEY

_model = ChatAnthropic(
    model=_settings.claude_model,
    api_key=SecretStr(_settings.anthropic_api_key),
    max_tokens=1024,
)

# Retries after the first attempt, for a TransientToolError. With the mock
# services' failure rates, 3 attempts in all turn a 40% per-call failure
# into ~6% (0.4^3), and 30% into ~3%.
TOOL_MAX_RETRIES = 2


def _transient_failure_message(exc: Exception) -> str:
    # What the model sees once retries are exhausted: enough to tell the
    # customer something useful, without the exception's internal detail.
    return (
        "This service is temporarily unavailable and didn't respond after several attempts. "
        "Tell the customer you couldn't complete this right now and suggest trying again shortly."
    )


def _unexpected_tool_error(exc: Exception, request: ToolCallRequest) -> str:
    # Anything that isn't a TransientToolError: a bug, the database being
    # down. Only the exception's type reaches the model — its message can
    # carry internal detail (a SQL statement, a connection string). The full
    # error is already in the JSONL trace, written by tools.py's wrapper.
    return (
        f"This tool failed unexpectedly ({type(exc).__name__}). "
        "Tell the customer you couldn't complete this right now; don't retry it."
    )


def tool_middleware(initial_delay: float = 0.5) -> list[AgentMiddleware[Any, Any, Any]]:
    """Retries transient tool failures, then turns every tool failure into a
    message for the model instead of an exception that ends the run.

    Order matters: the first middleware wraps the ones after it. The error
    handler sits outside the retries, so it only sees what's left once
    retrying has either succeeded, given up (and returned its own message),
    or passed through an exception it doesn't retry.
    """
    return [
        ToolErrorMiddleware(on_error=_unexpected_tool_error),
        ToolRetryMiddleware(
            max_retries=TOOL_MAX_RETRIES,
            retry_on=(TransientToolError,),
            on_failure=_transient_failure_message,
            initial_delay=initial_delay,
            backoff_factor=2.0,
            max_delay=4.0,
            jitter=True,
        ),
    ]


_TOOLS_BY_NAME = {t.name: t for t in LANGCHAIN_TOOLS}


def select_tools(allowed_tools: Sequence[str] | None) -> tuple[str, ...]:
    """The tool names a run may use: all of them when allowed_tools is None,
    otherwise exactly the allowlist. Raises ValueError on an unknown name
    rather than silently ignoring it — a typo in an allowlist should fail
    loudly, not quietly grant or drop access."""
    if allowed_tools is None:
        return tuple(_TOOLS_BY_NAME)
    unknown = sorted(set(allowed_tools) - _TOOLS_BY_NAME.keys())
    if unknown:
        raise ValueError(f"unknown tool(s) in allowed_tools: {', '.join(unknown)}")
    return tuple(name for name in _TOOLS_BY_NAME if name in allowed_tools)


def tools_for_scopes(scopes: Iterable[str]) -> tuple[str, ...]:
    """The tools a token's scopes grant. A scope this service doesn't know is
    ignored rather than rejected: a token can carry scopes meant for other
    services. A tool whose scope is missing is never given to the agent."""
    granted = set(scopes)
    return tuple(name for name in _TOOLS_BY_NAME if TOOL_SCOPES[name] in granted)


@lru_cache
def _get_agent(tool_names: tuple[str, ...]) -> Any:
    # Tool scoping is enforced by only giving the agent the allowed tools at
    # all — Claude never sees the others' schemas, so it can't call them.
    # One compiled graph per distinct allowlist, cached, since building one
    # isn't free and the set of allowlists in practice is small.
    tools = [_TOOLS_BY_NAME[name] for name in tool_names]
    return create_agent(
        model=_model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        context_schema=AgentContext,
        middleware=tool_middleware(),
    )


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            return str(block.get("text", ""))
    return "(no text response)"


async def run_agent(
    user_message: str,
    context: AgentContext,
    on_step: OnStep | None = None,
    allowed_tools: Sequence[str] | None = None,
) -> str:
    """context says who is asking (customer and store). It's the caller's
    job to establish it — from the authenticated request, never from the
    user's message — and it reaches tools only through ToolRuntime."""
    agent = _get_agent(select_tools(allowed_tools))
    reset_call_sequence()

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]},
            {"recursion_limit": RECURSION_LIMIT},
            context=context,
        )
    except GraphRecursionError:
        # The loop never produced a final answer within RECURSION_LIMIT
        # super-steps (see the comment on it above).
        trace_max_iterations(iterations=RECURSION_LIMIT)
        return "(stopped: hit the graph recursion limit without a final answer)"
    except Exception as exc:
        trace_api_error(iteration=-1, error=str(exc))
        raise

    messages: list[BaseMessage] = result["messages"]

    # create_agent() doesn't expose a per-step callback, so this
    # reconstructs step-by-step console output after the fact by walking the
    # final message list — every tool call and tool result is already in
    # there, just not surfaced live. (Tool calls are traced as they actually
    # happen, inside tools.py — this loop is only for the CLI's printed
    # output.)
    if on_step is not None:
        for message in messages:
            if isinstance(message, AIMessage) and message.tool_calls:
                for call in message.tool_calls:
                    on_step(f"  -> calling {call['name']}({call['args']})")
            elif isinstance(message, ToolMessage):
                on_step(f"  <- result: {message.content}")

    text = _message_text(messages[-1])
    trace_final_answer(iteration=len(messages), text=text)
    return text
