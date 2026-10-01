import pytest

from agent_service.shared import tools_core
from agent_service.shared.exceptions import TransientToolError


@pytest.fixture
def no_simulated_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_service.shared.tools_core.random.random", lambda: 0.9)


def _lookup(order_id: str, customer_id: str = "cust-1001", store_id: str = "main-street-market") -> str:
    return tools_core.lookup_order_status(
        tools_core.LookupOrderStatusInput(order_id=order_id), customer_id=customer_id, store_id=store_id
    )


@pytest.mark.usefixtures("orders_database", "no_simulated_failures")
class TestLookupOrderStatus:
    def test_customer_sees_their_own_order(self) -> None:
        assert "status=in_transit" in _lookup("ORD-1042")

    def test_order_id_is_case_and_whitespace_insensitive(self) -> None:
        assert "status=in_transit" in _lookup("  ord-1042 ")

    def test_another_customers_order_looks_exactly_like_a_missing_one(self) -> None:
        # ORD-2003 exists but belongs to cust-2002. The reply must not reveal
        # that: same message as an ID that doesn't exist at all.
        someone_elses = _lookup("ORD-2003")
        missing = _lookup("ORD-9999")
        assert someone_elses == 'Error: no order found with ID "ORD-2003"'
        assert missing == 'Error: no order found with ID "ORD-9999"'
        assert "processing" not in someone_elses

    def test_own_order_at_another_store_is_not_found(self) -> None:
        # PNP-5001 belongs to cust-3003 at pine-and-peak; the same customer ID
        # asking through main-street-market must not see it.
        assert _lookup("PNP-5001", customer_id="cust-3003", store_id="main-street-market").startswith(
            "Error: no order found"
        )
        assert "ready_for_pickup" in _lookup("PNP-5001", customer_id="cust-3003", store_id="pine-and-peak")


def test_lookup_order_status_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # force the simulated failure branch (raised before any database access)
    monkeypatch.setattr("agent_service.shared.tools_core.random.random", lambda: 0.1)
    with pytest.raises(TransientToolError):
        _lookup("ORD-1042")


def _notify(key: str) -> str:
    message = tools_core.SendNotificationInput(message="hi")
    return tools_core.send_notification(message, customer_id="cust-1001", idempotency_key=key)


def _delivered(key: str) -> int:
    return sum(1 for k, _, _ in tools_core.deliveries() if k == key)


def test_send_notification_goes_to_the_given_customer(no_simulated_failures: None) -> None:
    assert _notify("key-success") == 'Notification sent to cust-1001: "hi"'
    assert _delivered("key-success") == 1


def test_send_notification_failure_before_sending_delivers_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_service.shared.tools_core.random.random", lambda: 0.1)
    with pytest.raises(TransientToolError, match="unavailable"):
        _notify("key-before")
    assert _delivered("key-before") == 0


def test_send_notification_same_key_is_delivered_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # Delivered, but the confirmation is lost; a retry with the same key
    # reports success without delivering again.
    monkeypatch.setattr("agent_service.shared.tools_core.random.random", lambda: 0.2)
    with pytest.raises(TransientToolError, match="confirmation"):
        _notify("key-ambiguous")
    assert _notify("key-ambiguous") == 'Notification sent to cust-1001: "hi"'
    assert _delivered("key-ambiguous") == 1
