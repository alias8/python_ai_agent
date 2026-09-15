"""Scratch example: plain dicts vs. TypedDict vs. a real object (Pydantic
model), to make concrete the question that came up debugging
langchain/agent.py — why `call["name"]` instead of `call.name`.

Not used anywhere else in the app. Run it directly and step through in the
IntelliJ debugger the same way you did with langchain/cli.py:
    python -m agent_service.example
"""

from typing import TypedDict

from pydantic import BaseModel

# --- 1. A plain dict --------------------------------------------------
#
# No structure enforced at all — any keys, any types, checked by nothing.
# Always accessed with [] (or .get()), never with dot notation.

def plain_dict_example() -> None:
    order: dict[str, str] = {"id": "ORD-1042", "status": "in_transit"}

    print(order["id"])  # "ORD-1042"
    print(order.get("carrier"))  # None — key doesn't exist, .get() is safe
    # print(order["carrier"])    # would raise KeyError — key doesn't exist
    # print(order.id)            # would raise AttributeError — dicts have no attributes

    order["carrier"] = "FedEx"  # dicts are mutable; this just works
    for key, value in order.items():
        print(f"  {key} -> {value}")


# --- 2. A TypedDict -----------------------------------------------------
#
# Same runtime object as #1 — still a plain dict, still [] to access it.
# The only difference is mypy now knows its shape, so it can catch a typo'd
# key or a wrong-typed value *statically*, the same way it would for a class
# — this is exactly the shape langchain_core.messages.tool.ToolCall uses for
# `call` on line 100 of langchain/agent.py.

class OrderDict(TypedDict):
    id: str
    status: str


def typed_dict_example() -> None:
    order: OrderDict = {"id": "ORD-1042", "status": "in_transit"}

    print(order["id"])  # still [] — TypedDict is type-checking only
    # print(order.id)          # would raise AttributeError at runtime, same as a plain dict
    # bad: OrderDict = {"id": "ORD-1042"}       # mypy error: missing "status"
    # order["statuss"] = "x"                    # mypy error: unknown key (typo caught statically)

    print(type(order))  # <class 'dict'> — proof there's no special OrderDict type at runtime


# --- 3. A real object (Pydantic model) -----------------------------------
#
# This is the case that *does* support dot notation — an actual class
# instance, not a dict. AIMessage/ToolMessage in langchain_core are this
# kind of object, which is why `message.content` and `message.tool_calls`
# on lines 99/102 of langchain/agent.py use dots while `call["name"]` on
# line 100 doesn't — two different kinds of object in the same loop.

class Order(BaseModel):
    id: str
    status: str


def object_example() -> None:
    order = Order(id="ORD-1042", status="in_transit")

    print(order.id)  # dot notation — this is a real attribute on a real instance
    # print(order["id"])  # would raise TypeError — Order isn't subscriptable

    print(type(order))  # <class '...Order'> — a real, distinct type, unlike TypedDict


if __name__ == "__main__":
    plain_dict_example()
    typed_dict_example()
    object_example()
