import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# One JSONL file per process run, so each session's trace is easy to find and
# doesn't get interleaved with a previous run's.
_TRACE_DIR = Path("traces")
_TRACE_DIR.mkdir(parents=True, exist_ok=True)
_TRACE_FILE = _TRACE_DIR / f"{datetime.now(UTC).isoformat().replace(':', '-').replace('.', '-')}.jsonl"
# Create the file up front (not just on first write) so the path printed at
# startup always exists, even for a run that never calls a tool.
_TRACE_FILE.touch()


def get_trace_file_path() -> Path:
    return _TRACE_FILE


# The entries of the run in progress, when someone asked for them with
# capture_trace(). A ContextVar rather than a global so concurrent requests
# each collect their own: LangGraph copies the context into the worker
# threads it runs sync tools on, and the copies all point at the same list.
_captured: ContextVar[list[dict[str, Any]] | None] = ContextVar("captured_trace", default=None)


@contextmanager
def capture_trace() -> Iterator[list[dict[str, Any]]]:
    """Collects every entry traced inside the block, as well as writing it to
    the file — so /chat can return what its own run did."""
    entries: list[dict[str, Any]] = []
    token = _captured.set(entries)
    try:
        yield entries
    finally:
        _captured.reset(token)


def without_error_detail(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The entries as they can be shown to the customer: an exception's
    message can carry internal detail (a SQL statement, a connection string),
    so only its type leaves the server — the same rule agent.py applies to
    what the model sees. The message stays in the trace file."""
    return [{key: value for key, value in entry.items() if key != "error"} for entry in entries]


def _write(entry: dict[str, Any]) -> None:
    stamped = {"timestamp": datetime.now(UTC).isoformat(), **entry}
    with _TRACE_FILE.open("a") as f:
        f.write(json.dumps(stamped) + "\n")
    captured = _captured.get()
    if captured is not None:
        captured.append(stamped)


def trace_tool_call(
    *,
    iteration: int,
    tool: str,
    input: Any,
    duration_ms: int,
    success: bool,
    attempts: int,
    output: str | None = None,
    error: str | None = None,
    error_type: str | None = None,
    tool_call_id: str | None = None,
) -> None:
    _write(
        {
            "type": "tool_call",
            "iteration": iteration,
            # The model's ID for this call. A retried call is traced once per
            # attempt, and every attempt carries the same ID.
            "tool_call_id": tool_call_id,
            "tool": tool,
            "input": input,
            "duration_ms": duration_ms,
            "success": success,
            "attempts": attempts,
            "output": output,
            "error": error,
            "error_type": error_type,
        }
    )


def trace_tool_retry(*, iteration: int, tool: str, attempt: int, error: str, backoff_ms: int) -> None:
    # Logs one retried attempt after a transient tool failure, before the
    # retry actually happens. The eventual outcome is still captured by the
    # single trace_tool_call() entry once the retry loop finishes.
    _write(
        {
            "type": "tool_retry",
            "iteration": iteration,
            "tool": tool,
            "attempt": attempt,
            "error": error,
            "backoff_ms": backoff_ms,
        }
    )


def trace_final_answer(*, iteration: int, text: str) -> None:
    _write({"type": "final_answer", "iteration": iteration, "text": text})


def trace_api_error(*, iteration: int, error: str) -> None:
    _write({"type": "api_error", "iteration": iteration, "error": error})


def trace_max_iterations(*, iterations: int) -> None:
    _write({"type": "max_iterations", "iterations": iterations})


def trace_loop_detected(*, iteration: int, calls: list[str]) -> None:
    _write({"type": "loop_detected", "iteration": iteration, "calls": calls})
