"""The agent loop, built with LangChain's create_agent().

Sibling of ai-agent-demo's src/langchain/agent.ts. The growing message
array, the tool_use/tool_result wiring, and the "loop until no more tool
calls" condition are all built into create_agent()'s underlying LangGraph
state machine — the framework tradeoff: much less code to write, but the
loop mechanics are no longer something you can point at and read.

No retry/backoff is added around tools — see the project README for what
LangChain does by default when a tool raises.
"""

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from pydantic import SecretStr

from ..shared.config import get_settings
from ..shared.trace import trace_api_error, trace_final_answer, trace_max_iterations
from .context import AgentContext
from .tools import LANGCHAIN_TOOLS, reset_call_sequence

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
    "When you are done, give a clear, concise final answer."
)

OnStep = Callable[[str], None]

_settings = get_settings()  # fail fast on a missing ANTHROPIC_API_KEY

_model = ChatAnthropic(
    model=_settings.claude_model,
    api_key=SecretStr(_settings.anthropic_api_key),
    max_tokens=1024,
)

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


@lru_cache
def _get_agent(tool_names: tuple[str, ...]) -> Any:
    # Tool scoping is enforced by only giving the agent the allowed tools at
    # all — Claude never sees the others' schemas, so it can't call them.
    # One compiled graph per distinct allowlist, cached, since building one
    # isn't free and the set of allowlists in practice is small.
    tools = [_TOOLS_BY_NAME[name] for name in tool_names]
    return create_agent(
        model=_model, tools=tools, system_prompt=SYSTEM_PROMPT, context_schema=AgentContext
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
    store_id: str,
    on_step: OnStep | None = None,
    allowed_tools: Sequence[str] | None = None,
) -> str:
    """store_id scopes the run to one store (tenant). It's the caller's job
    to establish it — from the authenticated request, never from the user's
    message — and it reaches tools only through AgentContext."""
    agent = _get_agent(select_tools(allowed_tools))
    reset_call_sequence()

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]},
            {"recursion_limit": RECURSION_LIMIT},
            context=AgentContext(store_id=store_id),
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
