from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient

from agent_service.langchain.agent import tools_for_scopes
from agent_service.langchain.context import AgentContext
from agent_service.main import app
from agent_service.shared.auth import ALL_SCOPES, mint_token


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def seen_by_agent(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replaces the agent run, recording what /chat passed it."""
    seen: dict[str, Any] = {}

    async def fake_run_agent(message: str, **kwargs: Any) -> str:
        seen.update(kwargs, message=message)
        return "mocked answer"

    monkeypatch.setattr("agent_service.main.run_agent", fake_run_agent)
    return seen


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


CUST_1001 = _auth(mint_token("cust-1001", "main-street-market"))


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_runs_the_agent_as_the_tokens_customer_and_store(
    client: TestClient, seen_by_agent: dict[str, Any]
) -> None:
    response = client.post("/chat", json={"message": "where is my order?"}, headers=CUST_1001)
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "mocked answer"
    assert "trace_file" in body
    assert body["trace"] == []  # the agent run is mocked out, so nothing was traced
    assert seen_by_agent["context"] == AgentContext(customer_id="cust-1001", store_id="main-street-market")
    # mint_token's default is every scope, so every tool
    assert seen_by_agent["allowed_tools"] == tools_for_scopes(ALL_SCOPES)


def test_chat_rejects_empty_message(client: TestClient) -> None:
    response = client.post("/chat", json={"message": ""}, headers=CUST_1001)
    assert response.status_code == 422


def test_chat_gives_the_agent_only_the_tools_the_token_grants(
    client: TestClient, seen_by_agent: dict[str, Any]
) -> None:
    guest = _auth(mint_token("guest-4004", "main-street-market", scopes=["kb:read"]))
    response = client.post("/chat", json={"message": "where is ORD-1042?"}, headers=guest)
    assert response.status_code == 200
    assert seen_by_agent["allowed_tools"] == ("search_knowledge_base",)


@pytest.mark.parametrize(
    ("field", "value"),
    [("store_id", "pine-and-peak"), ("customer_id", "cust-2002"), ("allowed_tools", ["send_notification"])],
)
def test_identity_or_tools_in_the_body_are_rejected_not_trusted(
    client: TestClient, field: str, value: Any
) -> None:
    response = client.post("/chat", json={"message": "hi", field: value}, headers=CUST_1001)
    assert response.status_code == 422


def test_chat_requires_a_token(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "hi"})
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


_VALID_CLAIMS = {
    "sub": "cust-2002",
    "store_id": "main-street-market",
    "scope": "orders:read kb:read notifications:send",
    "aud": "agent-service",
    "exp": 4102444800,  # 2100-01-01
}


def _forged(claims: dict[str, Any], secret: str = "test-jwt-secret-not-for-production-use") -> str:
    return jwt.encode({**_VALID_CLAIMS, **claims}, secret, algorithm="HS256")


@pytest.mark.parametrize(
    "token",
    [
        pytest.param("not-a-jwt", id="garbage"),
        pytest.param(_forged({}, secret="a-different-secret-the-attacker-made-up"), id="wrong-signature"),
        pytest.param(mint_token("cust-1001", "main-street-market", ttl=timedelta(seconds=-1)), id="expired"),
        pytest.param(_forged({"aud": "some-other-service"}), id="wrong-audience"),
        pytest.param(_forged({"store_id": "no-such-store"}), id="unknown-store"),
        pytest.param(jwt.encode(_VALID_CLAIMS, None, algorithm="none"), id="alg-none"),
    ],
)
def test_chat_rejects_bad_tokens(client: TestClient, seen_by_agent: dict[str, Any], token: str) -> None:
    response = client.post("/chat", json={"message": "hi"}, headers=_auth(token))
    assert response.status_code == 401
    assert seen_by_agent == {}  # the agent never ran


@pytest.mark.parametrize("claim", ["store_id", "scope"])
def test_token_missing_a_required_claim_is_rejected(client: TestClient, claim: str) -> None:
    claims = {key: value for key, value in _VALID_CLAIMS.items() if key != claim}
    token = jwt.encode(claims, "test-jwt-secret-not-for-production-use", algorithm="HS256")
    response = client.post("/chat", json={"message": "hi"}, headers=_auth(token))
    assert response.status_code == 401
