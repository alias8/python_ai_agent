"""Customer authentication: verifies the JWTs the storefront issues.

This service doesn't log customers in. The storefront does, and gives the
customer's browser a short-lived JWT signed with a secret the two services
share. /chat verifies it and takes the customer and store from its claims —
the only source of "who is asking" the agent trusts. Nothing the customer
types, and nothing the model generates, can change it.

Mint a token for local testing (there's no storefront here):

    python -m agent_service.shared.auth --customer cust-1001 --store main-street-market
"""

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt

from .config import get_settings

# Pinned, never read from the token: a token's own `alg` header must not be
# able to pick a weaker algorithm (e.g. "none") to verify it with.
_ALGORITHM = "HS256"

# What a token lets the agent do on the customer's behalf, one scope per
# tool (langchain/tools.py's TOOL_SCOPES). The storefront decides which a
# customer gets — a guest who hasn't signed in might get only kb:read — and
# puts them in the token's space-separated `scope` claim, the OAuth
# convention. The browser never chooses them.
ALL_SCOPES = ("orders:read", "kb:read", "notifications:send")


@dataclass(frozen=True)
class Principal:
    """The authenticated customer, the store they're shopping with, and what
    the storefront allowed the agent to do for them."""

    customer_id: str
    store_id: str
    scopes: frozenset[str]


class AuthError(Exception):
    pass


def verify_token(token: str) -> Principal:
    settings = get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[_ALGORITHM],
            audience=settings.jwt_audience,
            options={"require": ["exp", "sub", "aud", "store_id", "scope"]},
        )
    except jwt.PyJWTError as exc:
        # One message for every failure (bad signature, expired, wrong
        # audience, missing claim): the caller gets a 401 either way, and
        # the detail is for logs, not for someone probing tokens.
        raise AuthError(f"invalid token: {exc}") from exc
    return Principal(
        customer_id=str(claims["sub"]),
        store_id=str(claims["store_id"]),
        scopes=frozenset(str(claims["scope"]).split()),
    )


def mint_token(
    customer_id: str,
    store_id: str,
    scopes: Iterable[str] = ALL_SCOPES,
    ttl: timedelta = timedelta(minutes=30),
) -> str:
    """What the storefront does after a customer logs in. Here only for
    local testing and the test suite."""
    settings = get_settings()
    now = datetime.now(UTC)
    claims = {
        "sub": customer_id,
        "store_id": store_id,
        "scope": " ".join(scopes),
        "aud": settings.jwt_audience,
        "iat": now,
        "exp": now + ttl,
    }
    return jwt.encode(claims, settings.jwt_secret, algorithm=_ALGORITHM)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mint a customer JWT for local testing.")
    parser.add_argument("--customer", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument(
        "--scope", default=" ".join(ALL_SCOPES), help=f"space-separated, from: {' '.join(ALL_SCOPES)}"
    )
    parser.add_argument("--minutes", type=int, default=30)
    args = parser.parse_args()
    print(mint_token(args.customer, args.store, args.scope.split(), timedelta(minutes=args.minutes)))


if __name__ == "__main__":
    main()
