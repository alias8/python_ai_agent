"""A REPL entry point for exercising the agent loop interactively."""

import argparse
import asyncio

from ..shared.trace import get_trace_file_path
from .agent import run_agent
from .kb_corpus import store_ids


async def _main(store_id: str) -> None:
    print("AI agent demo (Python, LangChain version). Ask something about an order, our policies, or")
    print("notifying a user.")
    print('Try: "What is the status of order ORD-1042, and let user-88 know once you find out"')
    print('Type "exit" to quit.')
    print(f"Store: {store_id}")
    print(f"Tool call trace: {get_trace_file_path()}\n")

    while True:
        user_input = input("> ")
        if user_input.strip().lower() == "exit":
            break
        answer = await run_agent(user_input, store_id=store_id, on_step=print)
        print(f"\n{answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store",
        default="main-street-market",
        choices=sorted(store_ids()),
        help="the store (tenant) whose knowledge base the agent searches",
    )
    asyncio.run(_main(parser.parse_args().store))


if __name__ == "__main__":
    main()
