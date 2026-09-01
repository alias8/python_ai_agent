import pytest

from agent_service import tools
from agent_service.exceptions import TransientToolError


def test_lookup_order_status_known_order(monkeypatch: pytest.MonkeyPatch) -> None:
    # skip the simulated transient failure
    monkeypatch.setattr("agent_service.tools.random.random", lambda: 0.9)
    result = tools.lookup_order_status(tools.LookupOrderStatusInput(order_id="ORD-1042"))
    assert "in_transit" in result


def test_lookup_order_status_unknown_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_service.tools.random.random", lambda: 0.9)
    result = tools.lookup_order_status(tools.LookupOrderStatusInput(order_id="ORD-9999"))
    assert result.startswith("Error:")


def test_lookup_order_status_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # force the simulated failure branch
    monkeypatch.setattr("agent_service.tools.random.random", lambda: 0.1)
    with pytest.raises(TransientToolError):
        tools.lookup_order_status(tools.LookupOrderStatusInput(order_id="ORD-1042"))


def test_search_knowledge_base_match() -> None:
    result = tools.search_knowledge_base(tools.SearchKnowledgeBaseInput(query="return policy"))
    assert "30 days" in result


def test_search_knowledge_base_no_match() -> None:
    result = tools.search_knowledge_base(tools.SearchKnowledgeBaseInput(query="spaceship warranty"))
    assert result.startswith("No knowledge base articles matched")


def test_send_notification_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_service.tools.random.random", lambda: 0.9)
    result = tools.send_notification(tools.SendNotificationInput(user_id="user-88", message="hi"))
    assert "sent to user-88" in result


def test_send_notification_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_service.tools.random.random", lambda: 0.1)
    with pytest.raises(TransientToolError):
        tools.send_notification(tools.SendNotificationInput(user_id="user-88", message="hi"))


def test_run_tool_dispatches_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_service.tools.random.random", lambda: 0.9)
    result = tools.run_tool("lookup_order_status", {"order_id": "ORD-1001"})
    assert "delivered" in result


def test_run_tool_unknown_tool() -> None:
    result = tools.run_tool("delete_everything", {})
    assert result.startswith('Error: unknown tool "delete_everything"')


def test_run_tool_disallowed_tool() -> None:
    result = tools.run_tool(
        "lookup_order_status", {"order_id": "ORD-1001"}, allowed_tools=["search_knowledge_base"]
    )
    assert "not permitted" in result


def test_run_tool_invalid_input_returns_error_string_not_exception() -> None:
    result = tools.run_tool("lookup_order_status", {})  # missing required order_id
    assert result.startswith('Error: invalid input for tool "lookup_order_status"')


def test_tool_definitions_schema_matches_input_model() -> None:
    names = {t["name"] for t in tools.TOOL_DEFINITIONS}
    assert names == {"lookup_order_status", "search_knowledge_base", "send_notification"}
    for definition in tools.TOOL_DEFINITIONS:
        assert definition["input_schema"]["type"] == "object"
        assert "properties" in definition["input_schema"]
