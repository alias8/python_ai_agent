"""A REPL entry point for exercising the agent loop interactively.

A local developer tool, so it takes the customer and store as flags instead
of a JWT — the HTTP API is where authentication happens."""

import argparse
import asyncio

from ..shared.trace import get_trace_file_path
from .agent import run_agent
from .context import AgentContext
from .kb_corpus import store_ids


async def _main(context: AgentContext) -> None:
    print("AI agent demo (Python, LangChain version). Ask something about an order, our policies, or")
    print("being notified.")
    print('Try: "What is the status of order ORD-1042? Send me a notification once you find out"')
    print('Type "exit" to quit.')
    print(f"Signed in as {context.customer_id} at {context.store_id}")
    print(f"Tool call trace: {get_trace_file_path()}\n")

    while True:
        user_input = input("> ")
        if user_input.strip().lower() == "exit":
            break
        answer = await run_agent(user_input, context=context, on_step=print)
        print(f"\n{answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store",
        default="main-street-market",
        choices=sorted(store_ids()),
        help="the store (tenant) whose knowledge base and orders the agent sees",
    )
    parser.add_argument(
        "--customer",
        default="cust-1001",
        help="the customer to act as (sample orders: cust-1001, cust-2002, cust-3003)",
    )
    args = parser.parse_args()
    asyncio.run(_main(AgentContext(customer_id=args.customer, store_id=args.store)))


if __name__ == "__main__":
    main()
