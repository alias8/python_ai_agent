import pytest

from agent_service.langchain.agent import select_tools

ALL_TOOLS = ("lookup_order_status", "search_knowledge_base", "send_notification")


def test_select_tools_defaults_to_every_tool() -> None:
    assert select_tools(None) == ALL_TOOLS


def test_select_tools_restricts_to_the_allowlist() -> None:
    assert select_tools(["search_knowledge_base"]) == ("search_knowledge_base",)


def test_select_tools_empty_allowlist_means_no_tools() -> None:
    assert select_tools([]) == ()


def test_select_tools_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="delete_everything"):
        select_tools(["search_knowledge_base", "delete_everything"])
