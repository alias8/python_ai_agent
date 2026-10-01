"""End to end through a real create_agent() run with a scripted model: the
customer the tools act for comes from the run's AgentContext, and nothing
the model sends can change it."""

from typing import Any
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from agent_service.langchain.context import AgentContext
from agent_service.langchain.tools import lookup_order_status_tool, send_notification_tool

CUST_1001 = AgentContext(customer_id="cust-1001", store_id="main-street-market")


class _ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":
        return self


async def _run_tool_call(tool: BaseTool, args: dict[str, Any], context: AgentContext) -> str:
    """Has a scripted model make exactly one call to `tool`, then returns
    what the tool sent back."""
    model = _ScriptedModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": tool.name, "args": args, "id": f"call_{uuid4().hex}"}],
                ),
                AIMessage(content="done"),
            ]
        )
    )
    agent = create_agent(model=model, tools=[tool], context_schema=AgentContext)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "hi"}]}, context=context)
    messages: list[BaseMessage] = result["messages"]
    return str(messages[2].content)


@pytest.fixture(autouse=True)
def no_simulated_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_service.shared.tools_core.random.random", lambda: 0.9)


def _schema_fields(tool: BaseTool) -> set[str]:
    schema = tool.tool_call_schema
    assert isinstance(schema, type) and issubclass(schema, BaseModel)
    return set(schema.model_json_schema()["properties"])


def test_the_model_cant_name_a_customer_or_recipient() -> None:
    assert _schema_fields(lookup_order_status_tool) == {"order_id"}
    assert _schema_fields(send_notification_tool) == {"message"}


@pytest.mark.usefixtures("orders_database")
async def test_forged_customer_argument_cant_unlock_another_customers_order() -> None:
    # A prompt-injected model asks for cust-2002's order, passing
    # cust-2002's ID as an argument. The argument is dropped; the lookup
    # runs as the context's customer, cust-1001, who doesn't own ORD-2003.
    output = await _run_tool_call(
        lookup_order_status_tool,
        {"order_id": "ORD-2003", "customer_id": "cust-2002", "store_id": "main-street-market"},
        CUST_1001,
    )
    assert output == 'Error: no order found with ID "ORD-2003"'


@pytest.mark.usefixtures("orders_database")
async def test_customer_can_look_up_their_own_order() -> None:
    output = await _run_tool_call(lookup_order_status_tool, {"order_id": "ORD-1042"}, CUST_1001)
    assert "status=in_transit" in output


async def test_notification_goes_to_the_signed_in_customer_whatever_the_model_asks() -> None:
    output = await _run_tool_call(
        send_notification_tool, {"message": "your order shipped", "user_id": "cust-2002"}, CUST_1001
    )
    assert output == 'Notification sent to cust-1001: "your order shipped"'
