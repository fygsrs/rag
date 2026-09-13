from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import get_current_user
from api.routes import sessions as sessions_route


class FakeMongoDBUtil:
    def __init__(self):
        self.cleared: list[str] = []

    def clear_session(self, session_id: str) -> int:
        self.cleared.append(session_id)
        return 2


def build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(sessions_route.router, prefix="/api/v1")
    return app


def test_delete_session_clears_scoped_conversation(monkeypatch):
    fake = FakeMongoDBUtil()
    monkeypatch.setattr(sessions_route, "get_mongodb_util", lambda: fake)
    app = build_app()
    app.dependency_overrides[get_current_user] = lambda: {"id": "user-1"}
    client = TestClient(app)

    response = client.delete("/api/v1/sessions/session-abc")

    assert response.status_code == 204
    assert fake.cleared == ["user-1:session-abc"]


def test_delete_session_requires_login(monkeypatch):
    fake = FakeMongoDBUtil()
    monkeypatch.setattr(sessions_route, "get_mongodb_util", lambda: fake)
    client = TestClient(build_app())

    response = client.delete("/api/v1/sessions/session-abc")

    assert response.status_code == 401
    assert fake.cleared == []
