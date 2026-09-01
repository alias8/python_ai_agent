"""The agentic loop: reason, call a tool, observe the result, repeat.

Python/FastAPI sibling of ai-agent-demo's direct-api implementation — same
loop shape and same production-readiness ports (retry/backoff, loop
detection, tool scoping, tracing), rebuilt with a Python-native tool stack
(Pydantic-validated tools) instead of the TypeScript one.
"""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable

from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam, TextBlock, ToolResultBlockParam, ToolUseBlock
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import get_settings
from .exceptions import TransientToolError
from .tools import TOOL_DEFINITIONS, run_tool
from .trace import (
    trace_api_error,
    trace_final_answer,
    trace_loop_detected,
    trace_max_iterations,
    trace_tool_call,
    trace_tool_retry,
)

MAX_ITERATIONS = 8
MAX_TOOL_RETRIES = 2  # up to 3 attempts total per tool call

OnStep = Callable[[str], None]

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant with access to a few tools: "
    "lookup_order_status, search_knowledge_base, and send_notification.\n"
    "Use tools whenever they would help answer the user's question accurately, and chain "
    "multiple tool calls together when a question requires more than one step (for example, "
    "looking up an order and then notifying the customer about it).\n"
    "When you are done, give a clear, concise final answer."
)


def _call_signature(block: ToolUseBlock) -> str:
    return f"{block.name}:{json.dumps(block.input, sort_keys=True)}"


async def _execute_tool_call(
    block: ToolUseBlock,
    iteration: int,
    allowed_tools: list[str] | None,
    on_step: OnStep | None,
) -> ToolResultBlockParam:
    # block.name/block.input are Claude's choice, not ours — it can also emit
    # more than one tool_use block per response (chained calls), which is why
    # the caller can invoke this more than once per turn.
    if on_step is not None:
        on_step(f"  -> calling {block.name}({json.dumps(block.input)})")

    start = time.monotonic()
    attempts = 0
    error: str | None = None
    result = ""

    def before_sleep(retry_state: RetryCallState) -> None:
        nonlocal attempts
        attempts = retry_state.attempt_number
        exc = retry_state.outcome.exception() if retry_state.outcome is not None else None
        backoff_s = retry_state.next_action.sleep if retry_state.next_action is not None else 0.0
        backoff_ms = int(backoff_s * 1000)
        trace_tool_retry(
            iteration=iteration, tool=block.name, attempt=attempts, error=str(exc), backoff_ms=backoff_ms
        )
        if on_step is not None:
            on_step(f"  (transient failure, retrying in {backoff_ms}ms: {exc})")

    # A tool can "fail" two ways: it raises TransientToolError, meaning the
    # (simulated) service itself is down and trying again might just work —
    # that's the only case worth retrying with backoff. Anything else — a
    # different exception, or a normal return with an error string like
    # run_tool's own not-permitted/unknown-tool message — is a "bad call"
    # that would fail identically every time, so it's returned as-is.
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(MAX_TOOL_RETRIES + 1),
            wait=wait_exponential(multiplier=0.1, min=0.1, max=2),
            retry=retry_if_exception_type(TransientToolError),
            before_sleep=before_sleep,
            reraise=True,
        ):
            with attempt:
                attempts = attempt.retry_state.attempt_number
                result = run_tool(block.name, dict(block.input), allowed_tools)
    except TransientToolError as exc:
        error = str(exc)
        result = f"Error: {error}"

    trace_tool_call(
        iteration=iteration,
        tool=block.name,
        input=block.input,
        duration_ms=int((time.monotonic() - start) * 1000),
        success=error is None and not result.startswith("Error"),
        attempts=attempts or 1,
        output=result,
        error=error,
    )

    if on_step is not None:
        on_step(f"  <- result: {result}")

    return {"type": "tool_result", "tool_use_id": block.id, "content": result}


async def run_agent(
    client: AsyncAnthropic,
    user_message: str,
    on_step: OnStep | None = None,
    allowed_tools: list[str] | None = None,
) -> str:
    messages: list[MessageParam] = [{"role": "user", "content": user_message}]
    previous_call_signatures: list[str] | None = None

    # Scoping which tools Claude is even offered — a run restricted to
    # ["search_knowledge_base"] never sees the other tools as options, so it
    # has no reason to ask for them. This is UX, not the real boundary: it's
    # enforced independently inside run_tool() (see _execute_tool_call above),
    # so a disallowed call is refused even if this filtering step were skipped.
    tools = (
        [t for t in TOOL_DEFINITIONS if t["name"] in allowed_tools]
        if allowed_tools is not None
        else TOOL_DEFINITIONS
    )

    settings = get_settings()

    # The API call itself is stateless — Claude has no memory of previous
    # requests, so the full conversation has to be resent every time. `messages`
    # is that running transcript: each loop iteration appends the assistant's
    # tool_use turn once the response arrives, then the tool_result turn once
    # the tool(s) have actually run, before calling client.messages.create()
    # again with the whole growing history.
    for iteration in range(MAX_ITERATIONS):
        response: Message
        try:
            response = await client.messages.create(
                model=settings.claude_model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )
        except Exception as exc:
            # Distinct from a tool call failing: this is the API call itself
            # (network, rate limit, auth) breaking, so the loop can't
            # continue. Trace it before rethrowing.
            trace_api_error(iteration=iteration, error=str(exc))
            raise

        messages.append({"role": "assistant", "content": response.content})

        # Claude itself decides whether a tool is needed, based on the tool
        # descriptions in `tools` above — nothing in this code inspects the
        # user's message.
        if response.stop_reason != "tool_use":
            text_block = next((b for b in response.content if isinstance(b, TextBlock)), None)
            text = text_block.text if text_block is not None else "(no text response)"
            trace_final_answer(iteration=iteration, text=text)
            return text

        tool_use_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]

        # Tools here are deterministic (aside from the simulated transient
        # failures), so if Claude wants to make the exact same call(s) it
        # just made, repeating them can only produce the same result again —
        # that's a stuck loop, not progress.
        call_signatures = sorted(_call_signature(b) for b in tool_use_blocks)
        if previous_call_signatures is not None and call_signatures == previous_call_signatures:
            trace_loop_detected(iteration=iteration, calls=call_signatures)
            if on_step is not None:
                on_step("  (stopped: Claude repeated the same tool call(s) as the previous step)")
            return "(stopped: detected a repeated tool call loop)"
        previous_call_signatures = call_signatures

        tool_call_tasks: list[Awaitable[ToolResultBlockParam]] = [
            _execute_tool_call(block, iteration, allowed_tools, on_step) for block in tool_use_blocks
        ]
        tool_results = list(await asyncio.gather(*tool_call_tasks))
        messages.append({"role": "user", "content": tool_results})

    trace_max_iterations(iterations=MAX_ITERATIONS)
    return "(stopped: hit max tool-call iterations without a final answer)"
