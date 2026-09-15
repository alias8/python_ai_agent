"""A REPL entry point equivalent to direct_api's CLI, for exercising the
LangChain implementation of the agent loop the same way."""

import asyncio

from ..shared.trace import get_trace_file_path
from .agent import run_agent


async def _main() -> None:
    print("AI agent demo (Python, LangChain version). Ask something about an order, our policies, or")
    print("notifying a user.")
    print('Try: "What is the status of order ORD-1042, and let user-88 know once you find out"')
    print('Type "exit" to quit.')
    print(f"Tool call trace: {get_trace_file_path()}\n")

    while True:
        user_input = input("> ")
        if user_input.strip().lower() == "exit":
            break
        answer = await run_agent(user_input, on_step=print)
        print(f"\n{answer}\n")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
