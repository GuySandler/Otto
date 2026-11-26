"""Per-user Google OAuth ("log in with Google" for the tools).

The server holds one OAuth *client* (from ``secrets/google_client.json`` or
``GOOGLE_CLIENT_ID``/``GOOGLE_CLIENT_SECRET``); each signed-in Otto account gets
its own refresh token, stored in the database. Nothing is global anymore: two
users on the same server have two separate Gmail accounts.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .integrations.google import SCOPES, scopes_for

# Google routinely grants an extra scope the client did not literally request
# (`openid` alongside `userinfo.email`, or anything already consented earlier via
# include_granted_scopes). oauthlib treats every such difference as a hard error
# ("Scope has changed from … to …"), which fails an otherwise perfect token
# exchange. Relaxing the check is oauthlib's documented escape hatch; we store the
# granted scopes anyway (`creds.scopes`) and the MCP servers enforce the real ones.
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

log = logging.getLogger("otto.google_oauth")

AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
USERINFO_URI = "https://openidconnect.googleapis.com/v1/userinfo"

#: how long a started-but-unfinished consent flow stays valid
STATE_TTL_S = 600


class GoogleOAuthError(RuntimeError):
    """Exchange failure that carries a short, human-facing reason."""

    def __init__(self, message: str, *, reason: str = "unknown") -> None:
        super().__init__(message)
        self.reason = reason


def explain_exchange_error(exc: BaseException) -> tuple[str, str]:
    """Map a Google/oauthlib failure to (short slug, sentence for the UI)."""
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if "scope has changed" in lowered:
        return (
            "scope_changed",
            "Google granted an extra scope and the OAuth library refused it. Otto now tolerates "
            "that — try Connect Google again.",
        )
    if "invalid_client" in lowered or "unauthorized_client" in lowered:
        return ("invalid_client", "Google rejected the client id/secret — check secrets/google_client.json.")
    if "redirect_uri_mismatch" in lowered or "redirect uri" in lowered:
        return (
            "redirect_mismatch",
            "The redirect URI must match byte for byte: set OTTO_PUBLIC_URL and register "
            "<that URL>/oauth/google/callback in the Google Cloud console.",
        )
    if "invalid_grant" in lowered:
        return ("invalid_grant", "That authorization code was already used or expired — try again.")
    if "invalid_scope" in lowered:
        return (
            "invalid_scope",
            "A requested scope is not declared on the OAuth consent screen (Data Access) — declare it "
            "or remove it from config/app.yaml → google.scopes.",
        )
    if "access_denied" in lowered:
        return ("access_denied", "The consent screen was declined.")
    return ("unknown", text[:300])


class GoogleOAuth:
    def __init__(self, settings, users) -> None:
        self.settings = settings
        self.users = users

    # ------------------------------------------------------------------ config
    def secrets_path(self) -> Path | None:
        """Where the OAuth client JSON actually is.

        Accepts the configured path, then the usual places a downloaded
        ``client_secret_*.json`` ends up: ``secrets/`` or the repo root. Resolved
        on every call so dropping the file in does not need a restart.
        """
        explicit = Path(self.settings.google_client_secrets_path)
        if explicit.exists():
            return explicit
        root = Path(self.settings.root)
        candidates: list[Path] = []
        for directory in (root / "secrets", root):
            if not directory.exists():
                continue
            candidates.extend(sorted(directory.glob("client_secret*.json")))
            candidates.append(directory / "google_client.json")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    @property
    def client_config(self) -> dict[str, Any] | None:
        path = self.secrets_path()
        if path is not None:
            try:
                data = json.loads(path.read_text())
            except ValueError:
                log.warning("google client secrets at %s are not valid JSON", path)
                data = {}
            if isinstance(data, dict):
                for key in ("web", "installed"):
                    if isinstance(data.get(key), dict):
                        return data
        client_id = str(getattr(self.settings, "google_client_id", "") or "").strip()
        client_secret = str(getattr(self.settings, "google_client_secret", "") or "").strip()
        if client_id and client_secret:
            return {
                "web": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": AUTH_URI,
                    "token_uri": TOKEN_URI,
                    "redirect_uris": [self.redirect_uri],
                }
            }
        return None

    @property
    def scopes(self) -> list[str]:
        return scopes_for(self.settings)

    @property
    def configured(self) -> bool:
        return self.client_config is not None

    @property
    def redirect_uri(self) -> str:
        return f"{self.settings.public_url.rstrip('/')}/oauth/google/callback"

    # ------------------------------------------------------------------- state
    def new_state(self, user_id: str) -> str:
        state = secrets.token_urlsafe(24)
        self.users.set_secret(user_id, f"oauth_state:{state[:12]}", json.dumps({"state": state, "at": time.time()}))
        return state

    def check_state(self, user_id: str, state: str) -> bool:
        record = self.users.get_secret(user_id, f"oauth_state:{str(state)[:12]}")
        if not record:
            return False
        try:
            data = json.loads(record)
        except ValueError:
            return False
        self.users.delete_secret(user_id, f"oauth_state:{state[:12]}")
        if data.get("state") != state:
            return False
        return (time.time() - float(data.get("at") or 0)) < STATE_TTL_S

    # --------------------------------------------------------------------- url
    def authorization_url(self, state: str) -> str:
        config = self.client_config
        if config is None:
            raise GoogleOAuthError("no Google OAuth client configured")
        section = config.get("web") or config.get("installed") or {}
        params = {
            "client_id": section.get("client_id", ""),
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.scopes),
            "access_type": "offline",
            "include_granted_scopes": "true",
            # force the consent screen so a refresh token is always issued
            "prompt": "consent",
            "state": state,
        }
        return f"{AUTH_URI}?{urlencode(params)}"

    # ---------------------------------------------------------------- exchange
    def exchange(self, code: str) -> tuple[str, str, list[str]]:
        """Trade the callback code for credentials. Returns (token_json, email, scopes)."""
        config = self.client_config
        if config is None:
            raise GoogleOAuthError("no Google OAuth client configured")
        from google_auth_oauthlib.flow import Flow

        section_key = "web" if "web" in config else "installed"
        flow = Flow.from_client_config(
            {section_key: config[section_key]},
            scopes=self.scopes,
            redirect_uri=self.redirect_uri,
        )
        try:
            flow.fetch_token(code=code)
        except Exception as exc:  # noqa: BLE001 - surfaced as a UI error
            reason, explanation = explain_exchange_error(exc)
            raise GoogleOAuthError(
                f"Google rejected the authorization code ({reason}): {explanation}", reason=reason
            ) from exc
        creds = flow.credentials
        email = ""
        try:
            import httpx

            response = httpx.get(
                USERINFO_URI,
                headers={"Authorization": f"Bearer {creds.token}"},
                timeout=10.0,
            )
            if response.status_code == 200:
                email = str(response.json().get("email") or "")
        except Exception:  # noqa: BLE001 - email is a nicety, not a requirement
            log.info("could not read the Google account email")
        scopes = list(creds.scopes or self.scopes)
        return creds.to_json(), email, scopes

    # ------------------------------------------------------------ last failure
    #: how long a failed connect stays visible in the UI
    ERROR_TTL_S = 900

    def set_error(self, user_id: str | None, reason: str, detail: str) -> None:
        if not user_id:
            return
        self.users.set_secret(
            user_id, "google_last_error", json.dumps({"reason": reason, "detail": detail, "at": time.time()})
        )

    def clear_error(self, user_id: str | None) -> None:
        if user_id:
            self.users.delete_secret(user_id, "google_last_error")

    def last_error(self, user_id: str | None) -> dict[str, Any] | None:
        if not user_id:
            return None
        raw = self.users.get_secret(user_id, "google_last_error")
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        if (time.time() - float(data.get("at") or 0)) > self.ERROR_TTL_S:
            return None
        return {"reason": str(data.get("reason") or ""), "detail": str(data.get("detail") or "")}

    # ------------------------------------------------------------------ status
    def status(self, user_id: str | None) -> dict[str, Any]:
        configured = self.configured
        record = self.users.get_google(user_id) if user_id else None
        connected = bool(record)
        if connected:
            detail = f"Connected as {record['email']}" if record.get("email") else "Connected"
        elif configured:
            detail = "Not connected"
        else:
            detail = "Add a Google OAuth client (docs/AUTH.md) to enable this"
        return {
            "configured": configured,
            "connected": connected,
            "email": (record or {}).get("email", ""),
            "scopes": (record or {}).get("scopes", []) or self.scopes,
            "detail": detail,
            "redirectUri": self.redirect_uri,
            "lastError": None if connected else self.last_error(user_id),
        }
