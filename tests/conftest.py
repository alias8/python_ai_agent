import os

# Settings() requires an API key at import time (it fails fast on a missing
# one, by design); the suite never calls Claude, so any value will do.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
