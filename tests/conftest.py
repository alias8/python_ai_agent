import os
from collections.abc import Callable
from typing import Any, cast

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

import pytest
from anthropic import AsyncAnthropic
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage


def _make_usage() -> Usage:
    return Usage(input_tokens=10, output_tokens=10)


def make_text_message(text: str, *, message_id: str = "msg_1") -> Message:
    return Message(
        id=message_id,
        content=[TextBlock(type="text", text=text)],
        model="claude-sonnet-5",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=_make_usage(),
    )


def make_tool_use_message(calls: list[tuple[str, dict[str, Any]]], *, message_id: str = "msg_1") -> Message:
    # Annotated as list[Any] rather than the inferred list[ToolUseBlock]: list
    # is invariant, so a concretely-typed list wouldn't satisfy Message's
    # content: list[<big Content union>] parameter even though every element
    # here is a valid member of that union.
    blocks: list[Any] = [
        ToolUseBlock(type="tool_use", id=f"toolu_{i}", name=name, input=tool_input)
        for i, (name, tool_input) in enumerate(calls)
    ]
    return Message(
        id=message_id,
        content=blocks,
        model="claude-sonnet-5",
        role="assistant",
        stop_reason="tool_use",
        stop_sequence=None,
        type="message",
        usage=_make_usage(),
    )


class FakeMessages:
    """Stands in for client.messages — returns one canned Message per call,
    in order, so a test can script a whole multi-turn agent run."""

    def __init__(self, responses: list[Message]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Message:
        # Snapshot the messages list rather than keeping the live reference:
        # run_agent keeps appending to the same list object after this call
        # returns, so storing kwargs as-is would make every past call's
        # "messages" silently reflect the final post-loop state instead of
        # what was actually sent at that point in time.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        try:
            return next(self._responses)
        except StopIteration as exc:
            raise AssertionError("FakeMessages ran out of canned responses") from exc


class FakeAnthropicClient:
    def __init__(self, responses: list[Message]) -> None:
        self.messages = FakeMessages(responses)


FakeClientFactory = Callable[[list[Message]], AsyncAnthropic]


@pytest.fixture
def fake_client_factory() -> FakeClientFactory:
    def _factory(responses: list[Message]) -> AsyncAnthropic:
        # Cast rather than typing run_agent's client param as the fake: the
        # loop only relies on the same narrow surface AsyncAnthropic exposes
        # (client.messages.create), so this is a test-boundary seam, not a
        # production type hole.
        return cast(AsyncAnthropic, FakeAnthropicClient(responses))

    return _factory
