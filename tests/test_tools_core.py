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


def test_send_notification_goes_to_the_given_customer(no_simulated_failures: None) -> None:
    message = tools_core.SendNotificationInput(message="hi")
    result = tools_core.send_notification(message, customer_id="cust-1001")
    assert result == 'Notification sent to cust-1001: "hi"'


def test_send_notification_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_service.shared.tools_core.random.random", lambda: 0.1)
    with pytest.raises(TransientToolError):
        tools_core.send_notification(tools_core.SendNotificationInput(message="hi"), customer_id="cust-1001")
