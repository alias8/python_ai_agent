from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent_service.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    # TestClient runs the app's lifespan (creates/tears down the Anthropic
    # client) via a context manager, matching how uvicorn would run it.
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_returns_agent_answer(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_agent(*_args: Any, **_kwargs: Any) -> str:
        return "mocked answer"

    monkeypatch.setattr("agent_service.main.run_agent", fake_run_agent)
    response = client.post("/chat", json={"message": "where is my order?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "mocked answer"
    assert "trace_file" in body


def test_chat_endpoint_rejects_empty_message(client: TestClient) -> None:
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_endpoint_passes_allowlist_to_agent(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_run_agent(_message: str, **kwargs: Any) -> str:
        seen.update(kwargs)
        return "mocked answer"

    monkeypatch.setattr("agent_service.main.run_agent", fake_run_agent)
    response = client.post(
        "/chat", json={"message": "hi", "allowed_tools": ["search_knowledge_base"]}
    )
    assert response.status_code == 200
    assert seen["allowed_tools"] == ["search_knowledge_base"]


def test_chat_endpoint_rejects_unknown_tool_in_allowlist(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "hi", "allowed_tools": ["delete_everything"]})
    assert response.status_code == 422
    assert "delete_everything" in response.json()["detail"]
