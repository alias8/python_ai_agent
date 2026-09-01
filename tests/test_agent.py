from typing import Any, cast

import pytest

from agent_service import agent as agent_module
from agent_service.agent import run_agent
from agent_service.exceptions import TransientToolError
from tests.conftest import FakeClientFactory, FakeMessages, make_text_message, make_tool_use_message


async def test_direct_text_reply_skips_tools(fake_client_factory: FakeClientFactory) -> None:
    client = fake_client_factory([make_text_message("hi there")])
    answer = await run_agent(client, "hello")
    assert answer == "hi there"


async def test_single_tool_call_then_final_answer(
    monkeypatch: pytest.MonkeyPatch, fake_client_factory: FakeClientFactory
) -> None:
    monkeypatch.setattr("agent_service.tools.random.random", lambda: 0.9)  # no simulated tool failure
    client = fake_client_factory(
        [
            make_tool_use_message([("lookup_order_status", {"order_id": "ORD-1001"})]),
            make_text_message("Your order has been delivered."),
        ]
    )
    answer = await run_agent(client, "where is my order ORD-1001?")
    assert answer == "Your order has been delivered."


async def test_disallowed_tool_is_refused_even_if_offered(
    monkeypatch: pytest.MonkeyPatch, fake_client_factory: FakeClientFactory
) -> None:
    # allowed_tools filters what Claude is *shown*, but run_tool() is the
    # real boundary — this simulates Claude asking for something outside its
    # allowlist anyway (e.g. via a bad prompt) and checks it's still refused.
    monkeypatch.setattr("agent_service.tools.random.random", lambda: 0.9)
    client = fake_client_factory(
        [
            make_tool_use_message([("send_notification", {"user_id": "u1", "message": "hi"})]),
            make_text_message("done"),
        ]
    )
    answer = await run_agent(client, "notify u1", allowed_tools=["lookup_order_status"])
    assert answer == "done"
    fake_messages = cast(FakeMessages, client.messages)
    tool_result = fake_messages.calls[1]["messages"][-1]["content"][0]
    assert "not permitted" in tool_result["content"]


async def test_loop_detected_on_repeated_identical_tool_call(
    monkeypatch: pytest.MonkeyPatch, fake_client_factory: FakeClientFactory
) -> None:
    monkeypatch.setattr("agent_service.tools.random.random", lambda: 0.9)
    repeated_call = make_tool_use_message([("lookup_order_status", {"order_id": "ORD-1001"})])
    client = fake_client_factory([repeated_call, repeated_call])
    answer = await run_agent(client, "where is my order?")
    assert "repeated tool call loop" in answer


async def test_hits_max_iterations_without_final_answer(
    monkeypatch: pytest.MonkeyPatch, fake_client_factory: FakeClientFactory
) -> None:
    monkeypatch.setattr("agent_service.tools.random.random", lambda: 0.9)
    # A different order ID each time, so loop detection doesn't fire first —
    # this is meant to exercise the iteration cap specifically.
    responses = [
        make_tool_use_message([("lookup_order_status", {"order_id": f"ORD-{i}"})])
        for i in range(agent_module.MAX_ITERATIONS)
    ]
    client = fake_client_factory(responses)
    answer = await run_agent(client, "keep checking orders")
    assert "hit max tool-call iterations" in answer


async def test_transient_tool_error_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, fake_client_factory: FakeClientFactory
) -> None:
    # Fail the first call, succeed on the retry — exercises the tenacity
    # retry path rather than the tool's own random simulation.
    call_count = {"n": 0}

    def flaky_random_tool(
        _name: str, _raw_input: dict[str, Any], _allowed_tools: list[str] | None = None
    ) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise TransientToolError("order service temporarily unavailable")
        return "order ORD-1001: delivered"

    monkeypatch.setitem(agent_module.__dict__, "run_tool", flaky_random_tool)
    client = fake_client_factory(
        [
            make_tool_use_message([("lookup_order_status", {"order_id": "ORD-1001"})]),
            make_text_message("Delivered."),
        ]
    )
    answer = await run_agent(client, "where is my order?")
    assert answer == "Delivered."
    assert call_count["n"] == 2
