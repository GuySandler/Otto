"""HTTP + websocket surface, exercised through Starlette's TestClient (no network)."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest
from starlette.testclient import TestClient

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

EMAIL = "api-test@example.com"
PASSWORD = "correct horse battery"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # keep the test database out of the repo's data dir
    db_path = tmp_path_factory.mktemp("db") / "otto-test.db"
    os.environ["OTTO_DB_PATH"] = str(db_path)
    os.environ.pop("LUNAROUTE_API_KEY", None)
    os.environ["OTTO_MCP_ENABLED"] = "0"   # offline tests

    from otto.main import app

    with TestClient(app) as test_client:
        state = test_client.app.state.otto
        state.llm = FakeLLMForApi()
        state.tts.settings.tts_enabled = False
        # a throwaway account for the whole module
        response = test_client.post("/api/auth/signup", json={"email": EMAIL, "password": PASSWORD, "name": "API Tester"})
        assert response.status_code == 200, response.text
        yield test_client
        created = Path(state.settings.data_dir) / "personalities" / "pytest-temp.yaml"
        created.unlink(missing_ok=True)
    os.environ.pop("OTTO_DB_PATH", None)
    os.environ.pop("OTTO_MCP_ENABLED", None)


class FakeLLMForApi:
    configured = True

    def __init__(self) -> None:
        from otto.pipeline.llm import StreamDone, TextDelta

        self._events = [TextDelta("Hello from the test."), StreamDone(text="Hello from the test.")]
        self._background = False

    def background_variant(self, model: str) -> str:
        return f"{model}-background" if self._background else model

    def has_background_variant(self, model: str) -> bool:
        return self._background

    async def stream(self, messages, *, model, tools=None, temperature=0.7, max_tokens=500, extra_body=None):
        for event in self._events:
            yield event

    async def complete(self, messages, *, model, max_tokens=16, temperature=0.0):
        return "APPROVE"

    async def models(self, *, refresh: bool = False):
        return [{"id": "fake-model", "label": "fake-model", "reasoning": False}]

    async def transcription_models(self, *, refresh: bool = False):
        return [{"id": "fake-stt", "label": "fake-stt"}]


@pytest.fixture
def anon(client):
    """Same app, but with the session cookie removed (signed out)."""
    saved = dict(client.cookies)
    client.cookies.clear()
    yield client
    client.cookies.clear()
    for key, value in saved.items():
        client.cookies.set(key, value)


# --------------------------------------------------------------------- auth
def test_root_redirects_to_login_when_signed_out(anon):
    response = anon.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/login"


def test_login_page_is_public(anon):
    response = anon.get("/login", follow_redirects=False)
    assert response.status_code in (200, 500)  # 500 only if web/login.html is missing


def test_api_requires_auth(anon):
    assert anon.get("/api/bootstrap").status_code == 401
    assert anon.get("/api/tools").status_code == 401
    assert anon.post("/api/config", json={}).status_code == 401


def test_websocket_requires_auth(anon):
    with pytest.raises(Exception):
        with anon.websocket_connect("/ws") as ws:
            ws.receive_text()


def test_auth_me_returns_user(client):
    body = client.get("/api/auth/me").json()
    assert body["user"]["email"] == EMAIL
    assert body["user"]["initials"]


def test_login_rejects_bad_password(anon):
    response = anon.post("/api/auth/login", json={"email": EMAIL, "password": "nope"})
    assert response.status_code == 401
    assert "Wrong email or password" in response.json()["detail"]


def test_signup_validates_input(anon):
    assert anon.post("/api/auth/signup", json={"email": "nope", "password": "longenough1"}).status_code == 400
    assert anon.post("/api/auth/signup", json={"email": f"{uuid.uuid4().hex}@x.co", "password": "short"}).status_code == 400
    duplicate = anon.post("/api/auth/signup", json={"email": EMAIL, "password": PASSWORD})
    assert duplicate.status_code == 409


def test_login_logout_roundtrip(client):
    saved = dict(client.cookies)
    client.cookies.clear()
    try:
        assert client.get("/api/auth/me").status_code == 401
        assert client.post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD}).status_code == 200
        assert client.get("/api/auth/me").status_code == 200
        assert client.post("/api/auth/logout").json()["ok"] is True
        assert client.get("/api/auth/me").status_code == 401
    finally:
        client.cookies.clear()
        client.post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD})
        del saved


# ------------------------------------------------------------------- surface
def test_health_is_public(anon):
    body = anon.get("/api/health").json()
    assert body["ok"] is True
    assert body["auth"]["signedIn"] is False


def test_bootstrap_shape(client):
    body = client.get("/api/bootstrap").json()
    assert body["app"]["name"]
    assert body["user"]["email"] == EMAIL
    assert isinstance(body["models"], list) and body["models"]
    assert any(p["id"] == "otto" for p in body["personalities"])
    assert {"google", "slack"} <= set(body["connections"])
    for key in ("model", "voice", "personality"):
        assert body["config"][key]


def test_bootstrap_exposes_capabilities(client):
    body = client.get("/api/bootstrap").json()
    caps = body["capabilities"]
    assert caps["parallel_agents"] is True
    assert caps["jobs"] is True
    assert set(caps["background_model"]) == {"available", "id"}


def test_connections_are_not_connected_by_default(client):
    body = client.get("/api/connections").json()
    assert body["google"]["connected"] is False
    assert body["slack"]["connected"] is False
    assert body["google"]["redirectUri"].endswith("/oauth/google/callback")


def test_slack_token_validation(client):
    assert client.post("/api/connections/slack", json={"token": ""}).status_code == 400
    assert client.post("/api/connections/slack", json={"token": "not-a-token"}).status_code == 400


def test_config_toggles_roundtrip(client):
    response = client.post("/api/config", json={"parallel_agents": True, "background_model": True})
    config = response.json()["config"]
    assert config["parallel_agents"] is True and config["background_model"] is True
    client.post("/api/config", json={"parallel_agents": False, "background_model": False})
    after = client.get("/api/bootstrap").json()["config"]
    assert after["parallel_agents"] is False and after["background_model"] is False


def test_tool_catalog_is_grouped(client):
    body = client.get("/api/tools").json()
    assert "system" in body["categories"]
    assert any(t["requires_confirmation"] for t in body["tools"])
    gmail = [t for t in body["tools"] if t["category"] == "gmail"]
    if body.get("mcp", {}).get("strict"):
        # strict MCP: the legacy REST tools are not offered at all
        assert gmail == []
    else:
        assert gmail and all(not t["configured"] for t in gmail)


def test_personality_read_and_write(client):
    otto = client.get("/api/personalities/otto").json()
    assert "You are Otto" in otto["system_prompt"]
    payload = {
        "name": "Temp",
        "tagline": "temporary",
        "system_prompt": "You are Temp.",
        "voice": {"voice": "Ryan", "language": "English"},
        "speech": {"fillers": ["Hm."]},
    }
    saved = client.put("/api/personalities/pytest-temp", json=payload).json()
    assert saved["id"] == "pytest-temp" and saved["source"] == "user"
    assert client.get("/api/personalities/pytest-temp").json()["system_prompt"] == "You are Temp."


def test_unknown_personality_is_404(client):
    assert client.get("/api/personalities/nope").status_code == 404


def test_index_page_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Otto" in response.text


def test_websocket_hello_and_text_turn(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello"}))
        ready = json.loads(ws.receive_text())
        assert ready["type"] == "ready"
        assert ready["config"]["model"]
        ws.send_text(json.dumps({"type": "text", "text": "hi"}))
        seen = []
        for _ in range(40):
            message = ws.receive()
            if "text" in message and message["text"]:
                event = json.loads(message["text"])
                seen.append(event)
                if event["type"] == "assistant.message":
                    break
        kinds = [event["type"] for event in seen]
        assert "user.text" in kinds and "assistant.message" in kinds
        assert seen[-1]["text"] == "Hello from the test."


def test_websocket_ping_pong(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "ping"}))
        assert json.loads(ws.receive_text())["type"] == "pong"


def test_websocket_config_message(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello"}))
        ws.receive_text()
        ws.send_text(json.dumps({"type": "config", "personality": "sable"}))
        event = json.loads(ws.receive_text())
        assert event["type"] == "config"
        assert event["config"]["personality"] == "sable"


def test_actions_api_roundtrip(client):
    from otto.main import app

    state = client.app.state.otto
    user = state.users.by_email(EMAIL)[0]
    state.actions.clear(user.id)

    empty = client.get("/api/actions").json()
    assert empty["actions"] == [] and empty["counts"]["open"] == 0

    action = state.actions.record(
        user.id,
        kind="draft",
        title="Re: lease",
        subtitle="to sarah@acme.com",
        url="https://mail.google.com/mail/u/0/#drafts",
        ref="draft:d1",
        source="gmail_draft",
    )

    listed = client.get("/api/actions").json()
    assert [item["id"] for item in listed["actions"]] == [action.id]
    assert listed["counts"]["open"] == 1

    done = client.post(f"/api/actions/{action.id}", json={"status": "done"}).json()
    assert done["action"]["status"] == "done"
    assert done["counts"]["open"] == 0
    assert client.get("/api/actions?status=done").json()["counts"]["done"] == 1

    assert client.post(f"/api/actions/{action.id}", json={"status": "nope"}).status_code == 400
    assert client.post("/api/actions/missing", json={"status": "done"}).status_code == 404
    state.actions.clear(user.id)


def test_bootstrap_reports_open_actions(client):
    body = client.get("/api/bootstrap").json()
    assert "actions" in body and isinstance(body["actions"]["open"], int)


def test_mcp_selftest_without_google_says_so(client):
    result = client.post("/api/mcp/selftest").json()
    assert result["ok"] is False
    assert result["tokenReady"] is False


def test_dashboard_page_is_served(client):
    from pathlib import Path

    page = Path(__file__).resolve().parent.parent / "web" / "dashboard.html"
    if not page.exists():
        pytest.skip("web/dashboard.html not built yet")
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "Actions required" in response.text


def test_dashboard_requires_auth(client):
    saved = dict(client.cookies)
    client.cookies.clear()
    try:
        assert client.get("/dashboard", follow_redirects=False).status_code == 302
    finally:
        client.cookies.clear()
        for key, value in saved.items():
            client.cookies.set(key, value)


def test_delete_account_requires_password(client):
    response = client.request("DELETE", "/api/auth/account", json={"password": "wrong"})
    assert response.status_code == 401
    # still signed in afterwards
    assert client.get("/api/auth/me").status_code == 200


# --------------------------------------------------------------- isolation
def test_google_connection_is_per_user(client):
    from otto.main import app

    state = client.app.state.otto
    user = state.users.by_email(EMAIL)[0].id
    other = state.users.create_user(f"other-{uuid.uuid4().hex[:6]}@example.com", "another password")

    state.users.save_google(user, '{"refresh_token": "r", "client_id": "i", "client_secret": "s"}', email="me@example.com", scopes=["x"])

    mine = client.get("/api/connections").json()
    assert mine["google"]["connected"] is True
    assert mine["google"]["email"] == "me@example.com"

    theirs = state.integrations_for(state.users.get(other.id))
    assert theirs.status()["google"]["connected"] is False

    # and the tools follow the credentials
    ctx_mine = state.tool_context(state.users.get(user))
    ctx_theirs = state.tool_context(other)
    schema_names = [s["function"]["name"] for s in state.registry.schemas(ctx=ctx_mine)]
    assert "gmail_search" in schema_names
    assert "gmail_search" not in [s["function"]["name"] for s in state.registry.schemas(ctx=ctx_theirs)]

    # disconnecting hides them again
    client.delete("/api/connections/google")
    assert client.get("/api/connections").json()["google"]["connected"] is False
