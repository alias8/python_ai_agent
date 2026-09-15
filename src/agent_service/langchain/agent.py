"""The same job as direct_api/agent.py, built with LangChain's create_agent()
instead of a hand-rolled loop.

Sibling of ai-agent-demo's src/langchain/agent.ts — see that file's comments
for the fuller comparison. The short version: this replaces essentially all
of direct_api/agent.py. The growing message array, the tool_use/tool_result
wiring, and the "loop until no more tool calls" condition are all built into
create_agent()'s underlying LangGraph state machine. What direct_api/agent.py
spells out by hand happens here in the two lines building `_agent` and the
one line calling `_agent.invoke(...)` below — the real framework tradeoff:
dramatically less code to write, but the loop mechanics are no longer
something you can point at and read.

None of direct_api's other production-readiness ports (retry/backoff, tool
scoping) are reimplemented here on purpose, to keep the comparison focused
on what the framework gives you unmodified — see the project README for
what LangChain provides by default instead.
"""

from collections.abc import Callable

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from pydantic import SecretStr

from ..shared.config import get_settings
from ..shared.trace import trace_api_error, trace_final_answer, trace_max_iterations
from .tools import LANGCHAIN_TOOLS, reset_call_sequence

# LangGraph's own default, not directly comparable to direct_api's
# MAX_ITERATIONS=8 — LangGraph counts each graph "super-step" (roughly: one
# model call, or one batch of tool executions), not one full
# model-call-then-tools round trip the way the hand-rolled loop counts, so
# the same number means something different here.
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

_agent = create_agent(model=_model, tools=LANGCHAIN_TOOLS, system_prompt=SYSTEM_PROMPT)


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            return str(block.get("text", ""))
    return "(no text response)"


async def run_agent(user_message: str, on_step: OnStep | None = None) -> str:
    reset_call_sequence()

    try:
        result = await _agent.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]},
            {"recursion_limit": RECURSION_LIMIT},
        )
    except GraphRecursionError:
        # Analogous to direct_api hitting MAX_ITERATIONS, though — per the
        # comment above — "recursion_limit" isn't counting the same unit.
        trace_max_iterations(iterations=RECURSION_LIMIT)
        return "(stopped: hit the graph recursion limit without a final answer)"
    except Exception as exc:
        trace_api_error(iteration=-1, error=str(exc))
        raise

    messages: list[BaseMessage] = result["messages"]

    # create_agent() doesn't expose a per-step callback the way the
    # hand-rolled loop's on_step parameter does, so this reconstructs the
    # same step-by-step console output after the fact by walking the final
    # message list — every tool call and tool result is already in there,
    # just not surfaced live. (Tool calls are traced as they actually
    # happen, inside tools.py — this loop is only about matching
    # direct_api's printed console output.)
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
