"""Users, sessions, per-user secrets and the Google OAuth state machine."""
from __future__ import annotations

import time

import pytest

from otto.db import Database
from otto.settings import Settings
from otto.users import (
    AuthError,
    SESSION_TTL_S,
    UserGoogleTokenStore,
    UserStore,
    hash_password,
    normalise_email,
    verify_password,
)


@pytest.fixture
def users(tmp_path):
    return UserStore(Database(tmp_path / "otto.db"))


def test_password_hash_roundtrip():
    stored = hash_password("hunter2hunter2")
    assert stored.startswith("scrypt$")
    assert "hunter2hunter2" not in stored
    assert verify_password("hunter2hunter2", stored) is True
    assert verify_password("hunter2hunter3", stored) is False
    # two hashes of the same password differ (per-user salt)
    assert hash_password("hunter2hunter2") != stored


def test_verify_password_survives_garbage():
    assert verify_password("x", "not-a-hash") is False
    assert verify_password("x", "scrypt$bad") is False


def test_email_normalisation():
    assert normalise_email("  Guy@Example.COM ") == "guy@example.com"
    with pytest.raises(AuthError):
        normalise_email("nope")


def test_create_user_and_login(users):
    user = users.create_user("a@b.co", "longenough1", "Ada Lovelace")
    assert user.initials == "AL"
    assert users.count() == 1

    logged_in, token = users.login("A@B.CO", "longenough1")
    assert logged_in.id == user.id
    assert users.user_for_token(token).email == "a@b.co"

    with pytest.raises(AuthError) as excinfo:
        users.login("a@b.co", "wrong")
    assert excinfo.value.status == 401


def test_signup_rules(users):
    users.create_user("dup@b.co", "longenough1")
    with pytest.raises(AuthError) as dup:
        users.create_user("dup@b.co", "longenough1")
    assert dup.value.status == 409
    with pytest.raises(AuthError):
        users.create_user("short@b.co", "tiny")


def test_sessions_expire(users):
    user = users.create_user("e@b.co", "longenough1")
    token = users.new_session(user.id)
    assert users.user_for_token(token) is not None
    users.db.execute("UPDATE sessions SET expires_at = ?", (time.time() - 1,))
    assert users.user_for_token(token) is None
    # the stale row is cleaned up on access
    assert users.db.query("SELECT * FROM sessions") == []


def test_logout_drops_only_that_session(users):
    user = users.create_user("s@b.co", "longenough1")
    first = users.new_session(user.id)
    second = users.new_session(user.id)
    users.drop_session(first)
    assert users.user_for_token(first) is None
    assert users.user_for_token(second) is not None


def test_rate_limit_after_repeated_failures(users):
    users.create_user("r@b.co", "longenough1")
    for _ in range(8):
        with pytest.raises(AuthError):
            users.login("r@b.co", "wrong")
    with pytest.raises(AuthError) as blocked:
        users.login("r@b.co", "longenough1")
    assert blocked.value.status == 429


def test_per_user_secrets_are_isolated(users):
    a = users.create_user("a1@b.co", "longenough1")
    b = users.create_user("b1@b.co", "longenough1")
    users.set_secret(a.id, "slack_bot_token", "xoxb-a")
    assert users.get_secret(a.id, "slack_bot_token") == "xoxb-a"
    assert users.get_secret(b.id, "slack_bot_token") is None
    users.delete_secret(a.id, "slack_bot_token")
    assert users.get_secret(a.id, "slack_bot_token") is None


def test_google_token_store_adapter(users):
    user = users.create_user("g@b.co", "longenough1")
    store = UserGoogleTokenStore(users, user.id)
    assert store.load() is None
    store.save('{"refresh_token": "r"}')
    assert store.load() == '{"refresh_token": "r"}'

    users.save_google(user.id, '{"refresh_token": "r2"}', email="g@b.co", scopes=["s1"])
    assert store.load() == '{"refresh_token": "r2"}'
    assert store.email == "g@b.co"

    store.clear()
    assert store.load() is None
    assert users.get_google(user.id) is None


def test_secrets_file_is_discovered_in_several_places(tmp_path):
    from otto.google_oauth import GoogleOAuth

    settings = Settings(root=tmp_path)
    oauth = GoogleOAuth(settings, StubUsers())
    assert oauth.configured is False

    # a downloaded client_secret_*.json dropped in the repo root is picked up
    (tmp_path / "client_secret_123-abc.apps.googleusercontent.com.json").write_text(
        '{"web": {"client_id": "cid", "client_secret": "secret"}}'
    )
    assert oauth.configured is True
    assert oauth.secrets_path().name.startswith("client_secret_")

    # ...and so is the canonical secrets/google_client.json
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "google_client.json").write_text('{"web": {"client_id": "c2", "client_secret": "s2"}}')
    assert oauth.secrets_path().name == "google_client.json"


def test_token_store_without_a_user_is_inert():
    store = UserGoogleTokenStore(None, None)  # type: ignore[arg-type]
    assert store.load() is None
    store.save("{}")  # no crash
    store.clear()


def test_delete_user_wipes_everything(users):
    user = users.create_user("d@b.co", "longenough1")
    users.set_secret(user.id, "slack_bot_token", "xoxb-x")
    users.save_google(user.id, '{"refresh_token": "r"}', email="d@b.co")
    token = users.new_session(user.id)

    with pytest.raises(AuthError):
        users.delete_user(user.id, "wrong-password")

    users.delete_user(user.id, "longenough1")
    assert users.count() == 0
    assert users.get(user.id) is None
    assert users.user_for_token(token) is None
    assert users.get_google(user.id) is None
    assert users.get_secret(user.id, "slack_bot_token") is None


class StubUsers(UserStore):
    """UserStore without a database, for OAuth-state tests."""

    def __init__(self) -> None:
        self._secrets: dict[tuple[str, str], str] = {}

    def set_secret(self, user_id, key, value):  # type: ignore[override]
        self._secrets[(user_id, key)] = value

    def get_secret(self, user_id, key):  # type: ignore[override]
        return self._secrets.get((user_id, key))

    def delete_secret(self, user_id, key):  # type: ignore[override]
        self._secrets.pop((user_id, key), None)

    def get_google(self, user_id):  # type: ignore[override]
        return None


def test_oauth_state_roundtrip(tmp_path):
    from otto.google_oauth import GoogleOAuth

    settings = Settings(root=tmp_path, google_client_id="cid", google_client_secret="secret")
    oauth = GoogleOAuth(settings, StubUsers())
    assert oauth.configured is True
    assert oauth.redirect_uri.endswith("/oauth/google/callback")

    state = oauth.new_state("u1")
    assert oauth.check_state("u1", state) is True
    # single use
    assert oauth.check_state("u1", state) is False
    assert oauth.check_state("u1", oauth.new_state("u1")) is True
    assert oauth.check_state("u1", "forged") is False


def test_oauth_url_contains_required_parameters(tmp_path):
    from otto.google_oauth import GoogleOAuth

    settings = Settings(
        root=tmp_path,
        google_client_id="cid",
        google_client_secret="secret",
        public_url_override="https://otto.test",
    )
    oauth = GoogleOAuth(settings, StubUsers())
    url = oauth.authorization_url("xyz")
    assert url.startswith("https://accounts.google.com/o/oauth2/auth?")
    assert "access_type=offline" in url
    assert "prompt=consent" in url
    assert "state=xyz" in url
    assert "redirect_uri=https%3A%2F%2Fotto.test%2Foauth%2Fgoogle%2Fcallback" in url


def test_last_error_is_recorded_and_expires(tmp_path):
    from otto.google_oauth import GoogleOAuth

    users = UserStore(Database(tmp_path / "otto.db"))
    user = users.create_user("e@b.co", "longenough1")
    oauth = GoogleOAuth(Settings(root=tmp_path, google_client_id="cid", google_client_secret="secret"), users)

    assert oauth.last_error(user.id) is None
    oauth.set_error(user.id, "scope_changed", "Google granted an extra scope")
    recorded = oauth.last_error(user.id)
    assert recorded["reason"] == "scope_changed"
    assert "extra scope" in recorded["detail"]
    assert oauth.status(user.id)["lastError"]["reason"] == "scope_changed"

    # a successful connect clears it
    oauth.clear_error(user.id)
    assert oauth.last_error(user.id) is None

    # stale errors disappear on their own
    oauth.set_error(user.id, "invalid_grant", "code reused")
    users.set_secret(
        user.id,
        "google_last_error",
        '{"reason": "invalid_grant", "detail": "code reused", "at": 1}',
    )
    assert oauth.last_error(user.id) is None


def test_exchange_errors_are_explained(tmp_path):
    from otto.google_oauth import explain_exchange_error

    cases = {
        "Scope has changed from 'a' to 'a openid'": "scope_changed",
        "invalid_client: Unauthorized": "invalid_client",
        "redirect_uri_mismatch": "redirect_mismatch",
        "invalid_grant: Bad Request": "invalid_grant",
        "invalid_scope": "invalid_scope",
        "access_denied": "access_denied",
        "something else": "unknown",
    }
    for message, expected in cases.items():
        reason, explanation = explain_exchange_error(Exception(message))
        assert reason == expected, message
        assert explanation


def test_oauth_relaxes_scope_matching():
    """Google appends `openid`; oauthlib must not fail the exchange over it."""
    import os

    import otto.google_oauth  # noqa: F401  (sets the env var at import)

    assert os.environ.get("OAUTHLIB_RELAX_TOKEN_SCOPE") == "1"


def test_oauth_not_configured(tmp_path):
    from otto.google_oauth import GoogleOAuth, GoogleOAuthError

    # a clean root, so the repo's own secrets/ cannot be discovered
    oauth = GoogleOAuth(Settings(root=tmp_path), StubUsers())
    assert oauth.configured is False
    with pytest.raises(GoogleOAuthError):
        oauth.authorization_url("x")
    status = oauth.status("u1")
    assert status["configured"] is False and status["connected"] is False
    assert "docs/AUTH.md" in status["detail"]
