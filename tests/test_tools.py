import pytest

from agent_service.direct_api import tools


def test_run_tool_dispatches_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_service.shared.tools_core.random.random", lambda: 0.9)
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
