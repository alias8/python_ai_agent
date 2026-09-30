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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt

from .config import get_settings

# Pinned, never read from the token: a token's own `alg` header must not be
# able to pick a weaker algorithm (e.g. "none") to verify it with.
_ALGORITHM = "HS256"


@dataclass(frozen=True)
class Principal:
    """The authenticated customer, and the store they're shopping with."""

    customer_id: str
    store_id: str


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
            options={"require": ["exp", "sub", "aud", "store_id"]},
        )
    except jwt.PyJWTError as exc:
        # One message for every failure (bad signature, expired, wrong
        # audience, missing claim): the caller gets a 401 either way, and
        # the detail is for logs, not for someone probing tokens.
        raise AuthError(f"invalid token: {exc}") from exc
    return Principal(customer_id=str(claims["sub"]), store_id=str(claims["store_id"]))


def mint_token(customer_id: str, store_id: str, ttl: timedelta = timedelta(minutes=30)) -> str:
    """What the storefront does after a customer logs in. Here only for
    local testing and the test suite."""
    settings = get_settings()
    now = datetime.now(UTC)
    claims = {
        "sub": customer_id,
        "store_id": store_id,
        "aud": settings.jwt_audience,
        "iat": now,
        "exp": now + ttl,
    }
    return jwt.encode(claims, settings.jwt_secret, algorithm=_ALGORITHM)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mint a customer JWT for local testing.")
    parser.add_argument("--customer", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--minutes", type=int, default=30)
    args = parser.parse_args()
    print(mint_token(args.customer, args.store, timedelta(minutes=args.minutes)))


if __name__ == "__main__":
    main()
