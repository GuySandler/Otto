"""Slack client — Web API over a bot token.

All calls funnel through :meth:`SlackClient._call`, which posts form-encoded
payloads and translates Slack's ``ok=false`` bodies into ``IntegrationError``.
"""
from __future__ import annotations

import difflib
import logging
import re
from typing import Any

import httpx

from . import IntegrationError, NotConfigured

log = logging.getLogger("otto.integrations.slack")

API_BASE = "https://slack.com/api"
TIMEOUT_S = 20.0
PAGE_SIZE = 200

#: channel / group / DM ids are passed through resolve_channel untouched
_CHANNEL_ID = re.compile(r"^[CGD][A-Z0-9]{6,}$")


def _permalink(channel: str, ts: str) -> str:
    """Deep link for a message; ``slack.com/archives`` redirects to the workspace."""
    if not ts:
        return ""
    stamp = str(ts).replace(".", "")
    return f"https://slack.com/archives/{channel}/p{stamp}"


class SlackClient:
    def __init__(self, settings, token: str | None = None) -> None:
        self.settings = settings
        #: per-user token when running in server mode, env token otherwise
        self._token = str(token or "").strip()
        self._http: httpx.AsyncClient | None = None
        self._users_cache: list[dict[str, Any]] | None = None

    @property
    def token(self) -> str:
        return self._token or str(self.settings.slack_bot_token or "").strip()

    @property
    def configured(self) -> bool:
        return bool(self.token)

    # ------------------------------------------------------------------ plumbing
    @property
    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=TIMEOUT_S)
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def status(self) -> dict[str, Any]:
        if self.configured:
            return {"configured": True, "connected": True, "team": None, "detail": "Slack bot token is configured"}
        return {
            "configured": False,
            "connected": False,
            "team": None,
            "detail": "No Slack bot token — add one in Settings → Connections",
        }

    # ------------------------------------------------------------------ http seam
    async def _call(self, method: str, **payload: Any) -> dict[str, Any]:
        token = self.token
        if not token:
            raise NotConfigured("Slack bot token is missing — add one in Settings → Connections")
        url = f"{API_BASE}/{method}"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = await self._client.post(url, data=_form(payload), headers=headers)
        except httpx.HTTPError as exc:
            raise IntegrationError(f"Slack {method} request failed: {exc}") from exc
        if response.status_code >= 400:
            raise IntegrationError(
                f"Slack {method} failed with HTTP {response.status_code}: {_excerpt(response)}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise IntegrationError(
                f"Slack {method} returned a non-JSON body: {_excerpt(response)}"
            ) from exc
        if not isinstance(data, dict):
            raise IntegrationError(
                f"Slack {method} returned an unexpected body: {_excerpt(response)}"
            )
        if not data.get("ok"):
            error = str(data.get("error") or "unknown_error")
            raise IntegrationError(_slack_error(method, error, data))
        return data

    # ------------------------------------------------------------------ channels
    async def channels(self, limit: int = 200) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        cursor = ""
        while len(out) < limit:
            payload: dict[str, Any] = {
                "types": "public_channel,private_channel",
                "exclude_archived": True,
                "limit": min(PAGE_SIZE, max(1, limit - len(out))),
            }
            if cursor:
                payload["cursor"] = cursor
            data = await self._call("conversations.list", **payload)
            for channel in data.get("channels") or []:
                out.append(
                    {
                        "id": str(channel.get("id") or ""),
                        "name": str(channel.get("name") or ""),
                        "is_private": bool(channel.get("is_private")),
                        "is_im": bool(channel.get("is_im")),
                        "member_count": int(channel.get("num_members") or 0),
                        "topic": str((channel.get("topic") or {}).get("value") or ""),
                    }
                )
            cursor = str((data.get("response_metadata") or {}).get("next_cursor") or "")
            if not cursor:
                break
        return out[:limit]

    async def resolve_channel(self, name_or_id: str) -> str:
        value = str(name_or_id or "").strip()
        if not value:
            raise IntegrationError("a Slack channel name or id is required")
        if _CHANNEL_ID.match(value):
            return value
        wanted = value.lstrip("#").strip().lower()
        channels = await self.channels()
        by_name = {channel["name"].lower(): channel["id"] for channel in channels if channel["name"]}
        if wanted in by_name:
            return by_name[wanted]
        names = list(by_name)
        close = [name for name in names if wanted and (wanted in name or name in wanted)]
        if not close:
            close = difflib.get_close_matches(wanted, names, n=5, cutoff=0.6)
        message = f"Slack channel '{value}' not found"
        if close:
            message += "; close matches: " + ", ".join(f"#{name}" for name in close[:5])
        raise IntegrationError(message)

    # ------------------------------------------------------------------- messages
    async def history(
        self,
        channel: str,
        limit: int = 20,
        oldest: str | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"channel": channel, "limit": int(limit)}
        if oldest:
            payload["oldest"] = str(oldest)
        data = await self._call("conversations.history", **payload)
        messages = list(data.get("messages") or [])
        messages.reverse()  # Slack returns newest first; callers want oldest first
        names = await self._user_names()
        out: list[dict[str, Any]] = []
        for message in messages:
            text = str(message.get("text") or "")
            if message.get("subtype") == "bot_message" and not text.strip():
                continue
            user_id = str(message.get("user") or "")
            ts = str(message.get("ts") or "")
            out.append(
                {
                    "ts": ts,
                    "user": user_id,
                    "user_name": names.get(user_id) or str(message.get("username") or "") or user_id,
                    "text": text,
                    "thread_ts": str(message.get("thread_ts") or ""),
                    "replies": len(message.get("replies") or []),
                    "permalink": _permalink(channel, ts),
                }
            )
        return out

    async def search(self, query: str, count: int = 10) -> list[dict[str, Any]]:
        data = await self._call(
            "search.messages",
            query=query,
            count=int(count),
            sort="timestamp",
            sort_dir="desc",
        )
        matches = (data.get("messages") or {}).get("matches") or []
        names = await self._user_names()
        out: list[dict[str, Any]] = []
        for match in matches:
            channel = match.get("channel") or {}
            user_id = str(match.get("user") or "")
            out.append(
                {
                    "ts": str(match.get("ts") or ""),
                    "channel": str(channel.get("id") or ""),
                    "channel_name": str(channel.get("name") or ""),
                    "user": user_id,
                    "user_name": names.get(user_id) or str(match.get("username") or "") or user_id,
                    "text": str(match.get("text") or ""),
                    "permalink": str(match.get("permalink") or ""),
                }
            )
        return out

    async def post_message(
        self,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"channel": channel, "text": text, "unfurl_links": False}
        if thread_ts:
            payload["thread_ts"] = str(thread_ts)
        data = await self._call("chat.postMessage", **payload)
        return {
            "ok": bool(data.get("ok")),
            "ts": str(data.get("ts") or ""),
            "channel": str(data.get("channel") or channel),
        }

    # ---------------------------------------------------------------------- users
    async def users(self) -> list[dict[str, Any]]:
        if self._users_cache is not None:
            return self._users_cache
        out: list[dict[str, Any]] = []
        cursor = ""
        while True:
            payload: dict[str, Any] = {"limit": PAGE_SIZE}
            if cursor:
                payload["cursor"] = cursor
            data = await self._call("users.list", **payload)
            for member in data.get("members") or []:
                profile = member.get("profile") or {}
                out.append(
                    {
                        "id": str(member.get("id") or ""),
                        "name": str(member.get("name") or ""),
                        "real_name": str(profile.get("real_name") or member.get("real_name") or ""),
                        "email": str(profile.get("email") or ""),
                        "is_bot": bool(member.get("is_bot")),
                    }
                )
            cursor = str((data.get("response_metadata") or {}).get("next_cursor") or "")
            if not cursor:
                break
        self._users_cache = out
        return out

    async def _user_names(self) -> dict[str, str]:
        try:
            members = await self.users()
        except IntegrationError as exc:
            log.warning("could not resolve Slack user names: %s", exc)
            return {}
        return {
            member["id"]: (member["real_name"] or member["name"] or member["id"])
            for member in members
            if member["id"]
        }


# ------------------------------------------------------------------ payload helpers
def _form(payload: dict[str, Any]) -> dict[str, str]:
    """Form-encode values; Slacks wants lowercase booleans and no ``None`` keys."""
    out: dict[str, str] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = "true" if value else "false"
        else:
            out[key] = str(value)
    return out


def _excerpt(response: httpx.Response, limit: int = 300) -> str:
    text = " ".join(str(getattr(response, "text", "") or "").split())
    return text[:limit]


def _slack_error(method: str, error: str, data: Any) -> str:
    if error == "missing_scope":
        needed = ""
        if isinstance(data, dict):
            needed = str(data.get("needed") or "")
        if needed:
            return (
                f"Slack {method} failed: missing_scope — the bot token needs '{needed}'. "
                "Add it in the Slack app's OAuth scopes and reinstall the app."
            )
        return (
            f"Slack {method} failed: missing_scope — the bot token lacks a required OAuth scope. "
            "Check the app's OAuth scopes and reinstall it."
        )
    return f"Slack {method} failed: {error}"
