class TransientToolError(Exception):
    """Raised to simulate a flaky upstream service being temporarily down.

    Distinct from a "bad call" (e.g. an unknown order ID), which would fail
    identically every time and isn't worth retrying — only this error is
    something a retry with backoff can plausibly fix.
    """
