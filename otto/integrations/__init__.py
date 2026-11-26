"""Integration clients (Google, Slack) shared by the tool modules.

Clients are per-user: each signed-in account has its own Google refresh token and
optional Slack bot token, so one Otto server can serve several mailboxes without
them bleeding into each other. Everything is imported lazily so the app still
boots when a dependency or a credential is missing.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("otto.integrations")


class IntegrationError(RuntimeError):
    """Raised when an upstream API call fails."""


class NotConfigured(IntegrationError):
    """Raised when credentials for an integration are missing."""


class Integrations:
    """Lazily constructed container handed to every tool through ToolContext."""

    def __init__(self, settings, *, users: Any = None, user_id: str | None = None) -> None:
        self.settings = settings
        self.users = users
        self.user_id = user_id
        self._google: Any = None
        self._slack: Any = None

    # ------------------------------------------------------------------ google
    @property
    def google(self) -> Any:
        if self._google is None:
            from .google import GoogleClient, FileTokenStore

            if self.users is not None and self.user_id:
                from ..users import UserGoogleTokenStore

                store = UserGoogleTokenStore(self.users, self.user_id)
            else:
                store = FileTokenStore(self.settings)
            self._google = GoogleClient(self.settings, store=store)
        return self._google

    # ------------------------------------------------------------------- slack
    @property
    def slack(self) -> Any:
        if self._slack is None:
            from .slack import SlackClient

            token = None
            if self.users is not None and self.user_id:
                token = self.users.get_secret(self.user_id, "slack_bot_token")
            self._slack = SlackClient(self.settings, token=token)
        return self._slack

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in ("google", "slack"):
            try:
                out[name] = getattr(self, name).status()
            except Exception as exc:  # noqa: BLE001
                out[name] = {"configured": False, "connected": False, "detail": f"{type(exc).__name__}: {exc}"}
        return out

    async def aclose(self) -> None:
        for client in (self._google, self._slack):
            if client is not None:
                try:
                    await client.aclose()
                except Exception:  # noqa: BLE001
                    pass


def get_integrations(settings, *, users: Any = None, user_id: str | None = None) -> Integrations:
    return Integrations(settings, users=users, user_id=user_id)
