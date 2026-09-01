import json
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


def _write(entry: dict[str, Any]) -> None:
    line = json.dumps({"timestamp": datetime.now(UTC).isoformat(), **entry})
    with _TRACE_FILE.open("a") as f:
        f.write(line + "\n")


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
) -> None:
    _write(
        {
            "type": "tool_call",
            "iteration": iteration,
            "tool": tool,
            "input": input,
            "duration_ms": duration_ms,
            "success": success,
            "attempts": attempts,
            "output": output,
            "error": error,
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
