"""A REPL entry point equivalent to ai-agent-demo's direct-api CLI, for
exercising the agent loop without needing the HTTP server running."""

import asyncio

from anthropic import AsyncAnthropic

from .agent import run_agent
from .config import get_settings
from .trace import get_trace_file_path


async def _main() -> None:
    settings = get_settings()  # fail fast on a missing ANTHROPIC_API_KEY
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    print("AI agent demo (Python). Ask something about an order, our policies, or notifying a user.")
    print('Try: "What is the status of order ORD-1042, and let user-88 know once you find out"')
    print('Type "exit" to quit.')
    print(f"Tool call trace: {get_trace_file_path()}\n")

    try:
        while True:
            user_input = input("> ")
            if user_input.strip().lower() == "exit":
                break
            answer = await run_agent(client, user_input, on_step=print)
            print(f"\n{answer}\n")
    finally:
        await client.close()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
