"""A failing tool must never crash the agent run (and so /chat): transient
failures are retried, and whatever still fails reaches the model as a
message it can relay. Driven through the real run_agent with a scripted
model in place of Claude, and retry delays set to zero."""

from collections.abc import Iterator
from functools import partial
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agent_service.langchain import agent as agent_module
from agent_service.langchain.context import AgentContext
from agent_service.main import app
from agent_service.shared import tools_core
from agent_service.shared.auth import mint_token
from agent_service.shared.exceptions import TransientToolError

CUST_1001 = AgentContext(customer_id="cust-1001", store_id="main-street-market")


class _ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":
        return self


@pytest.fixture
def script_model(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Swaps Claude for a model that makes one given tool call, then answers "done".

    Each call gets a fresh ID, as real tool calls do — it doubles as the
    notification idempotency key, so a reused ID would look "already sent"."""

    def install(tool_name: str, args: dict[str, Any]) -> None:
        model = _ScriptedModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": tool_name, "args": args, "id": f"call_{uuid4().hex}"}],
                    ),
                    AIMessage(content="done"),
                ]
            )
        )
        monkeypatch.setattr(agent_module, "_model", model)
        agent_module._get_agent.cache_clear()

    no_delay = partial(agent_module.tool_middleware, initial_delay=0)
    monkeypatch.setattr(agent_module, "tool_middleware", no_delay)
    yield install
    agent_module._get_agent.cache_clear()


def _rolls(monkeypatch: pytest.MonkeyPatch, *values: float) -> None:
    """The simulated services' random draws, in order (then success forever)."""
    sequence = iter(values)
    monkeypatch.setattr("agent_service.shared.tools_core.random.random", lambda: next(sequence, 0.99))


async def _tool_results(message: str = "hi") -> tuple[str, list[str]]:
    steps: list[str] = []
    answer = await agent_module.run_agent(message, context=CUST_1001, on_step=steps.append)
    return answer, [step.removeprefix("  <- result: ") for step in steps if step.startswith("  <- result: ")]


@pytest.mark.usefixtures("orders_database")
async def test_transient_failures_are_retried_until_the_tool_succeeds(
    script_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_model("lookup_order_status", {"order_id": "ORD-1042"})
    _rolls(monkeypatch, 0.1, 0.1)  # fail, fail, then succeed on the last allowed attempt
    answer, results = await _tool_results()
    assert answer == "done"
    assert results == ["order ORD-1042: status=in_transit, carrier=FedEx, eta=2026-10-03"]


async def test_exhausted_retries_become_a_message_not_a_crash(
    script_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_model("lookup_order_status", {"order_id": "ORD-1042"})
    attempts = 0

    def always_down(*args: Any, **kwargs: Any) -> str:
        nonlocal attempts
        attempts += 1
        raise TransientToolError("order service temporarily unavailable")

    monkeypatch.setattr(tools_core, "lookup_order_status", always_down)
    monkeypatch.setattr("agent_service.langchain.tools.lookup_order_status", always_down)
    answer, results = await _tool_results()
    assert answer == "done"
    assert attempts == 1 + agent_module.TOOL_MAX_RETRIES
    assert results[0].startswith("This service is temporarily unavailable")


async def test_unexpected_errors_reach_the_model_without_their_internal_detail(
    script_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_model("lookup_order_status", {"order_id": "ORD-1042"})
    attempts = 0

    def broken(*args: Any, **kwargs: Any) -> str:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("connection to postgres://admin:hunter2@db failed")

    monkeypatch.setattr("agent_service.langchain.tools.lookup_order_status", broken)
    answer, results = await _tool_results()
    assert answer == "done"
    assert attempts == 1  # not a transient failure, so never retried
    assert "RuntimeError" in results[0]
    assert "hunter2" not in results[0]


async def test_notification_lost_confirmation_is_retried_without_sending_twice(
    script_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Attempt 1: delivered, then the confirmation is lost (roll 0.2). The
    # retry carries the same idempotency key, so it's recognized as already
    # delivered rather than sent again.
    script_model("send_notification", {"message": "your order shipped"})
    _rolls(monkeypatch, 0.2)
    before = len(tools_core.deliveries())
    answer, results = await _tool_results()
    assert answer == "done"
    assert results == ['Notification sent to cust-1001: "your order shipped"']
    new = tools_core.deliveries()[before:]
    assert [(customer, message) for _, customer, message in new] == [("cust-1001", "your order shipped")]


@pytest.mark.usefixtures("orders_database")
def test_chat_returns_200_when_the_order_service_is_down(
    script_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The original bug: a TransientToolError escaped the run and /chat 500'd.
    script_model("lookup_order_status", {"order_id": "ORD-1042"})
    # the order service never recovers
    monkeypatch.setattr("agent_service.shared.tools_core.random.random", lambda: 0.1)
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "where is ORD-1042?"},
            headers={"Authorization": f"Bearer {mint_token('cust-1001', 'main-street-market')}"},
        )
    assert response.status_code == 200
    assert response.json()["answer"] == "done"
