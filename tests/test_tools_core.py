import pytest

from agent_service.shared import tools_core
from agent_service.shared.exceptions import TransientToolError


def test_lookup_order_status_known_order(monkeypatch: pytest.MonkeyPatch) -> None:
    # skip the simulated transient failure
    monkeypatch.setattr("agent_service.shared.tools_core.random.random", lambda: 0.9)
    result = tools_core.lookup_order_status(tools_core.LookupOrderStatusInput(order_id="ORD-1042"))
    assert "in_transit" in result


def test_lookup_order_status_unknown_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_service.shared.tools_core.random.random", lambda: 0.9)
    result = tools_core.lookup_order_status(tools_core.LookupOrderStatusInput(order_id="ORD-9999"))
    assert result.startswith("Error:")


def test_lookup_order_status_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # force the simulated failure branch
    monkeypatch.setattr("agent_service.shared.tools_core.random.random", lambda: 0.1)
    with pytest.raises(TransientToolError):
        tools_core.lookup_order_status(tools_core.LookupOrderStatusInput(order_id="ORD-1042"))


def test_send_notification_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_service.shared.tools_core.random.random", lambda: 0.9)
    result = tools_core.send_notification(tools_core.SendNotificationInput(user_id="user-88", message="hi"))
    assert "sent to user-88" in result


def test_send_notification_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_service.shared.tools_core.random.random", lambda: 0.1)
    with pytest.raises(TransientToolError):
        tools_core.send_notification(tools_core.SendNotificationInput(user_id="user-88", message="hi"))
