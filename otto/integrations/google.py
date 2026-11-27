"""Google client — Gmail + Calendar REST over OAuth2 user credentials.

Every HTTP call goes through :meth:`GoogleClient._request`, which owns the whole
auth lifecycle: load the token file, refresh it when expired, retry once on 401
(after a forced refresh) and turn failures into ``IntegrationError``.
"""
from __future__ import annotations

import asyncio
import base64
import html
import json
import logging
import re
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import httpx
from google.auth import exceptions as google_auth_exceptions
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from . import IntegrationError, NotConfigured

log = logging.getLogger("otto.integrations.google")

#: default consent request: the scope set Google documents for its Workspace MCP
#: servers (GA, Nov 2026). config/app.yaml → google.scopes overrides this list;
#: declare whatever you ask for on the consent screen's Data Access page or
#: Google rejects it with invalid_scope. No `gmail.send` — the Gmail MCP server
#: has no send tool, so Otto drafts and the user presses send.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
    "https://www.googleapis.com/auth/calendar.events.freebusy",
    "https://www.googleapis.com/auth/calendar.events.readonly",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/userinfo.email",
]


def scopes_for(settings) -> list[str]:
    """Scopes to request, from `google.scopes` in the config (or the defaults)."""
    configured = settings.cfg("google", "scopes", default=None) if settings is not None else None
    if isinstance(configured, list) and configured:
        return [str(scope) for scope in configured]
    return list(SCOPES)

def _as_rfc3339(value: Any) -> str | None:
    """Coerce a timestamp to RFC3339 with an offset, or None if unusable.

    Google's Calendar API rejects a naive "2026-11-30T00:00:00" (no zone); a
    missing zone means *local time* to the person speaking, so attach the
    server's local offset.
    """
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.isoformat()


GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
CALENDAR_BASE = "https://www.googleapis.com/calendar/v3/calendars"

TIMEOUT_S = 30.0
#: cap on body_text handed to the model/UI
MAX_BODY_CHARS = 20000
#: concurrent metadata fetches when a search returns many ids
SEARCH_CONCURRENCY = 5
METADATA_HEADERS = ("From", "To", "Subject", "Date")

_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_BREAKING_TAG = re.compile(r"<\s*(br|/p|/div|/tr|/li)\s*/?\s*>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>]+>")


# --------------------------------------------------------------------------- auth
class FileTokenStore:
    """Single-user token persisted next to the OAuth client (CLI / legacy mode)."""

    def __init__(self, settings) -> None:
        self.path = Path(settings.google_token_file)

    def load(self) -> str | None:
        try:
            return self.path.read_text()
        except OSError:
            return None

    def save(self, token_json: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(token_json)
        self.path.chmod(0o600)

    def clear(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            return

    @property
    def email(self) -> str:
        return ""


def _authorized_json(token_json: str | None) -> bool:
    if not token_json:
        return False
    try:
        data = json.loads(token_json)
    except ValueError:
        return False
    return isinstance(data, dict) and bool(data.get("access_token") or data.get("refresh_token"))


def build_credentials(settings) -> Credentials:
    """Load the stored OAuth user credentials (no network I/O).

    Raises ``NotConfigured`` when the token file is missing or unreadable so the
    caller can tell the user to run ``python scripts/google_auth.py``.
    """
    token_path = Path(settings.google_token_file)
    if not token_path.exists():
        raise NotConfigured(
            f"Google token not found at {token_path} — run: python scripts/google_auth.py"
        )
    try:
        return Credentials.from_authorized_user_file(str(token_path), SCOPES)
    except (OSError, ValueError, KeyError) as exc:
        raise NotConfigured(
            f"Google token at {token_path} is unreadable ({exc}) — re-run: python scripts/google_auth.py"
        ) from exc


def _token_file_authorized(token_path: Path) -> bool:
    try:
        data = json.loads(Path(token_path).read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    return bool(data.get("access_token") or data.get("refresh_token"))


class GoogleClient:
    """Gmail + Calendar client bound to one :class:`otto.settings.Settings`."""

    def __init__(self, settings, store=None) -> None:
        self.settings = settings
        #: anything with load()/save()/clear() — a per-user DB store in server mode,
        #: a JSON file in single-user/CLI mode
        self.store = store if store is not None else FileTokenStore(settings)
        self.scopes = scopes_for(settings)
        self._http: httpx.AsyncClient | None = None
        self._creds: Credentials | None = None
        self._token_json: str | None = None
        self._creds_lock = asyncio.Lock()

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

    @property
    def connected(self) -> bool:
        """True when this client holds usable credentials (cheap, no network)."""
        try:
            return _authorized_json(self.store.load())
        except Exception:  # noqa: BLE001
            return False

    @property
    def account_email(self) -> str:
        return str(getattr(self.store, "email", "") or "")

    def status(self) -> dict[str, Any]:
        connected = self.connected
        email = self.account_email
        if connected:
            detail = f"Connected as {email}" if email else "Connected"
        else:
            detail = "Not connected — open Settings → Connections to sign in with Google"
        return {
            "configured": connected,
            "connected": connected,
            "authorized": connected,
            "email": email,
            "detail": detail,
            "scopes": list(self.scopes),
        }

    # -------------------------------------------------------------------- tokens
    async def _get_credentials(self) -> Credentials:
        token_json = await asyncio.to_thread(self.store.load)
        if self._creds is not None and token_json == self._token_json:
            return self._creds  # still matches what is stored
        if not token_json:
            raise NotConfigured("Google is not connected — open Settings → Connections to sign in.")
        try:
            info = json.loads(token_json)
            # Refresh with the scopes the user actually GRANTED, not the config
            # wishlist: Google rejects a refresh asking for scopes the OAuth
            # client does not declare (invalid_scope). No stored list → pass no
            # hint at all and Google keeps the originally granted set.
            self._creds = Credentials.from_authorized_user_info(info, None)
            self._token_json = token_json
        except (ValueError, KeyError) as exc:
            raise NotConfigured(f"Stored Google credentials are unreadable ({exc}) — reconnect Google.") from exc
        return self._creds

    async def access_token(self) -> str:
        """A fresh access token (refreshing if needed) — used by the MCP client."""
        return await self._ensure_token()

    async def _ensure_token(self, *, force: bool = False) -> str:
        """Return a usable access token, refreshing/persisting it when needed."""
        async with self._creds_lock:
            creds = await self._get_credentials()
            if not force and creds.valid and creds.token:
                return creds.token
            if not creds.refresh_token:
                raise NotConfigured("Google credentials have no refresh token — reconnect Google in Settings.")
            await self._refresh(creds)
            return str(creds.token or "")

    async def _refresh(self, creds: Credentials) -> None:
        def _refresh_and_persist() -> None:
            creds.refresh(Request())
            token_json = creds.to_json()
            self.store.save(token_json)
            self._token_json = token_json

        try:
            await asyncio.to_thread(_refresh_and_persist)
        except google_auth_exceptions.RefreshError as exc:
            raise NotConfigured(f"Google refused to refresh the token ({exc}) — reconnect Google.") from exc
        except OSError as exc:
            raise IntegrationError(f"could not persist the refreshed Google token: {exc}") from exc
        log.info("refreshed Google access token")

    # ------------------------------------------------------------------ http seam
    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Single HTTP seam for every Gmail/Calendar call (adds auth, retries 401 once)."""
        token = await self._ensure_token()
        response = await self._send(method, url, token, params=params, json=json)
        if response.status_code == 401:
            log.info("Google returned 401 for %s %s — refreshing token and retrying once", method, url)
            token = await self._ensure_token(force=True)
            response = await self._send(method, url, token, params=params, json=json)
        if response.status_code >= 400:
            raise IntegrationError(
                f"{method} {url} failed with HTTP {response.status_code}: {_excerpt(response)}"
            )
        return _decode_json(response, method, url)

    async def _send(
        self,
        method: str,
        url: str,
        token: str,
        *,
        params: dict[str, Any] | None,
        json: dict[str, Any] | None,
    ) -> httpx.Response:
        headers = {"Authorization": f"Bearer {token}"}
        try:
            return await self._client.request(method, url, params=params, json=json, headers=headers)
        except httpx.HTTPError as exc:
            raise IntegrationError(f"{method} {url} failed: {exc}") from exc

    # --------------------------------------------------------------------- Gmail
    async def gmail_search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        data = await self._request(
            "GET",
            f"{GMAIL_BASE}/messages",
            params={"q": query or "", "maxResults": int(max_results), "includeSpamTrash": "false"},
        )
        message_ids = [str(m.get("id")) for m in data.get("messages") or [] if m.get("id")]
        if not message_ids:
            return []
        semaphore = asyncio.Semaphore(SEARCH_CONCURRENCY)

        async def _fetch(message_id: str) -> dict[str, Any]:
            async with semaphore:
                return await self._gmail_metadata(message_id)

        return list(await asyncio.gather(*(_fetch(message_id) for message_id in message_ids)))

    async def _gmail_metadata(self, message_id: str) -> dict[str, Any]:
        data = await self._request(
            "GET",
            f"{GMAIL_BASE}/messages/{quote(message_id, safe='')}",
            params={"format": "metadata", "metadataHeaders": list(METADATA_HEADERS)},
        )
        payload = data.get("payload") or {}
        headers = _header_map(payload)
        labels = [str(label) for label in data.get("labelIds") or []]
        return {
            "id": str(data.get("id") or message_id),
            "thread_id": str(data.get("threadId") or ""),
            "from_": headers.get("from", ""),
            "to": headers.get("to", ""),
            "subject": headers.get("subject", ""),
            "snippet": str(data.get("snippet") or ""),
            "date": _message_date(headers, data.get("internalDate")),
            "unread": "UNREAD" in labels,
            "labels": labels,
        }

    async def gmail_get(self, message_id: str, format: str = "full") -> dict[str, Any]:
        data = await self._request(
            "GET",
            f"{GMAIL_BASE}/messages/{quote(message_id, safe='')}",
            params={"format": format},
        )
        payload = data.get("payload") or {}
        headers = _header_map(payload)
        labels = [str(label) for label in data.get("labelIds") or []]
        return {
            "id": str(data.get("id") or message_id),
            "thread_id": str(data.get("threadId") or ""),
            "from_": headers.get("from", ""),
            "to": headers.get("to", ""),
            "cc": headers.get("cc", ""),
            "subject": headers.get("subject", ""),
            "snippet": str(data.get("snippet") or ""),
            "date": _message_date(headers, data.get("internalDate")),
            "body_text": _extract_body(payload),
            "labels": labels,
            "unread": "UNREAD" in labels,
            "attachments": _attachments(payload),
        }

    async def gmail_draft(
        self,
        to: str | Iterable[str],
        subject: str,
        body: str,
        cc: str | Iterable[str] = (),
        bcc: str | Iterable[str] = (),
        html: bool = False,
    ) -> dict[str, Any]:
        raw = _build_raw(to, subject, body, cc=cc, bcc=bcc, html=html)
        data = await self._request(
            "POST", f"{GMAIL_BASE}/drafts", json={"message": {"raw": raw}}
        )
        message = data.get("message") or {}
        return {
            "id": str(data.get("id") or ""),
            "message_id": str(message.get("id") or ""),
        }

    async def gmail_send(
        self,
        to: str | Iterable[str],
        subject: str,
        body: str,
        cc: str | Iterable[str] = (),
        bcc: str | Iterable[str] = (),
        html: bool = False,
    ) -> dict[str, Any]:
        raw = _build_raw(to, subject, body, cc=cc, bcc=bcc, html=html)
        data = await self._request("POST", f"{GMAIL_BASE}/messages/send", json={"raw": raw})
        return {
            "id": str(data.get("id") or ""),
            "thread_id": str(data.get("threadId") or ""),
        }

    async def gmail_labels(self) -> list[dict[str, Any]]:
        data = await self._request("GET", f"{GMAIL_BASE}/labels")
        return [
            {
                "id": str(label.get("id") or ""),
                "name": str(label.get("name") or ""),
                "type": str(label.get("type") or ""),
            }
            for label in data.get("labels") or []
        ]

    # ------------------------------------------------------------------ Calendar
    def _calendar_events_url(self, calendar_id: str | None) -> str:
        calendar = str(calendar_id or self.settings.gcal_calendar_id or "primary")
        return f"{CALENDAR_BASE}/{quote(calendar, safe='')}/events"

    # ------------------------------------------------------------- people api
    async def people_profile(self) -> dict[str, Any]:
        """The user's own name and email — used to sign drafted email."""
        data = await self._request(
            "GET",
            "https://people.googleapis.com/v1/people/me",
            params={"personFields": "names,emailAddresses"},
        )
        names = data.get("names") or []
        display = ""
        for entry in names:
            display = str(entry.get("displayName") or "").strip()
            if display:
                break
        email = ""
        for entry in data.get("emailAddresses") or []:
            email = str(entry.get("value") or "").strip()
            if email:
                break
        return {"name": display, "email": email or str(self.account_email or "")}

    async def people_search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Search the user's Google contacts by name; returns name + email."""
        data = await self._request(
            "GET",
            "https://people.googleapis.com/v1/people/me/connections",
            params={
                "query": str(query or "").strip(),
                "personFields": "names,emailAddresses",
                "pageSize": max(1, min(int(limit), 10)),
            },
        )
        out: list[dict[str, Any]] = []
        for person in data.get("connections") or []:
            name = ""
            for entry in person.get("names") or []:
                name = str(entry.get("displayName") or "").strip()
                if name:
                    break
            emails = [str(e.get("value")) for e in (person.get("emailAddresses") or []) if e.get("value")]
            if name or emails:
                out.append({"name": name or "(unnamed)", "email": emails[0] if emails else "",
                            "emails": emails})
        return out

    async def calendar_list(
        self,
        time_min: str | None = None,
        time_max: str | None = None,
        max_results: int = 20,
        query: str | None = None,
        calendar_id: str | None = None,
    ) -> list[dict[str, Any]]:
        now = datetime.now().astimezone()
        # Google's API requires an offset; voice models often send none ("the
        # 30th at midnight") — a naive stamp means *local* time
        start = _as_rfc3339(time_min) or now.isoformat(timespec="seconds")
        end = _as_rfc3339(time_max) or (now + timedelta(days=7)).isoformat(timespec="seconds")
        params: dict[str, Any] = {
            "timeMin": start,
            "timeMax": end,
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": int(max_results),
        }
        if query:
            params["q"] = str(query)
        data = await self._request("GET", self._calendar_events_url(calendar_id), params=params)
        return [_event_shape(event) for event in data.get("items") or []]

    async def calendar_get(self, event_id: str, calendar_id: str | None = None) -> dict[str, Any]:
        url = f"{self._calendar_events_url(calendar_id)}/{quote(str(event_id), safe='')}"
        return _event_shape(await self._request("GET", url))

    async def calendar_create(
        self,
        summary: str,
        start: str,
        end: str,
        description: str = "",
        location: str = "",
        attendees: Iterable[str] = (),
        timezone: str | None = None,
        calendar_id: str | None = None,
        reminders: Iterable[dict[str, Any] | None] | None = None,
        use_default_reminders: bool | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "summary": summary,
            "start": _event_time(start, timezone),
            "end": _event_time(end, timezone),
        }
        extra = _reminders_body(reminders, use_default_reminders)
        if extra is not None:
            body["reminders"] = extra
        if description:
            body["description"] = description
        if location:
            body["location"] = location
        emails = _split_addresses(attendees)
        if emails:
            body["attendees"] = [{"email": email} for email in emails]
        data = await self._request("POST", self._calendar_events_url(calendar_id), json=body)
        return _event_shape(data)

    async def calendar_patch(
        self,
        event_id: str,
        *,
        summary: str | None = None,
        start: str | None = None,
        end: str | None = None,
        description: str | None = None,
        location: str | None = None,
        attendees: Iterable[str] | None = None,
        timezone: str | None = None,
        calendar_id: str | None = None,
        reminders: Iterable[dict[str, Any] | None] | None = None,
        use_default_reminders: bool | None = None,
    ) -> dict[str, Any]:
        """Move / rename / re-remind an existing event (events.patch, only the fields given)."""
        body: dict[str, Any] = {}
        extra = _reminders_body(reminders, use_default_reminders)
        if extra is not None:
            body["reminders"] = extra
        if summary:
            body["summary"] = summary
        if start:
            body["start"] = _event_time(start, timezone)
        if end:
            body["end"] = _event_time(end, timezone)
        if description is not None and description != "":
            body["description"] = description
        if location is not None and location != "":
            body["location"] = location
        emails = _split_addresses(attendees or ())
        if emails:
            body["attendees"] = [{"email": email} for email in emails]
        if not body:
            raise IntegrationError("nothing to update")
        url = f"{self._calendar_events_url(calendar_id)}/{quote(str(event_id), safe='')}"
        data = await self._request("PATCH", url, json=body)
        return _event_shape(data)

    async def calendar_delete(self, event_id: str, calendar_id: str | None = None) -> dict[str, Any]:
        url = f"{self._calendar_events_url(calendar_id)}/{quote(str(event_id), safe='')}"
        await self._request("DELETE", url)
        return {"id": str(event_id), "deleted": True}


# ------------------------------------------------------------------ payload helpers
def _excerpt(response: httpx.Response, limit: int = 300) -> str:
    text = " ".join(str(getattr(response, "text", "") or "").split())
    return text[:limit]


def _decode_json(response: httpx.Response, method: str, url: str) -> dict[str, Any]:
    if not (response.content or b"").strip():
        return {}  # 204 from events.delete
    try:
        data = response.json()
    except ValueError as exc:
        raise IntegrationError(
            f"{method} {url} returned non-JSON body: {_excerpt(response)}"
        ) from exc
    return data if isinstance(data, dict) else {"items": data}


def _header_map(payload: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for header in payload.get("headers") or []:
        name = str(header.get("name") or "").lower()
        if name and name not in out:
            out[name] = str(header.get("value") or "")
    return out


def _message_date(headers: dict[str, str], internal_date: Any) -> str:
    """Prefer the RFC 2822 ``Date`` header, fall back to Gmail's internal epoch ms."""
    header = headers.get("date", "").strip()
    if header:
        return header
    try:
        stamp = int(str(internal_date)) / 1000
    except (TypeError, ValueError):
        return ""
    return datetime.fromtimestamp(stamp).astimezone().isoformat(timespec="seconds")


def _b64url_decode(value: str) -> str:
    data = str(value or "")
    if not data:
        return ""
    padding = "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(data + padding).decode("utf-8", "replace")
    except (ValueError, TypeError):
        return ""


def _iter_parts(payload: dict[str, Any]):
    yield payload
    for part in payload.get("parts") or []:
        yield from _iter_parts(part)


def _extract_body(payload: dict[str, Any], limit: int = MAX_BODY_CHARS) -> str:
    """Walk the MIME tree preferring text/plain, else a stripped text/html fallback."""
    plain: list[str] = []
    rich: list[str] = []
    for part in _iter_parts(payload):
        mime = str(part.get("mimeType") or "").lower()
        data = _b64url_decode((part.get("body") or {}).get("data") or "")
        if not data.strip():
            continue
        if mime == "text/plain":
            plain.append(data)
        elif mime == "text/html":
            rich.append(data)
    text = "\n".join(plain).strip()
    if not text:
        text = _html_to_text("\n".join(rich))
    return text[:limit]


def _html_to_text(markup: str) -> str:
    text = _SCRIPT_STYLE.sub(" ", markup)
    text = _BREAKING_TAG.sub("\n", text)
    text = _ANY_TAG.sub("", text)
    text = html.unescape(text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _attachments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for part in _iter_parts(payload):
        filename = str(part.get("filename") or "")
        if not filename:
            continue
        out.append(
            {
                "filename": filename,
                "size": int((part.get("body") or {}).get("size") or 0),
                "mime": str(part.get("mimeType") or ""),
            }
        )
    return out


def _split_addresses(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = [str(item) for item in value]
    return [item.strip() for item in items if str(item).strip()]


def _build_raw(
    to: str | Iterable[str],
    subject: str,
    body: str,
    *,
    cc: str | Iterable[str] = (),
    bcc: str | Iterable[str] = (),
    html: bool = False,
) -> str:
    message = EmailMessage()
    message["To"] = ", ".join(_split_addresses(to))
    cc_list = _split_addresses(cc)
    if cc_list:
        message["Cc"] = ", ".join(cc_list)
    bcc_list = _split_addresses(bcc)
    if bcc_list:
        message["Bcc"] = ", ".join(bcc_list)
    message["Subject"] = subject
    message.set_content(body, subtype="html" if html else "plain")
    return base64.urlsafe_b64encode(message.as_bytes()).decode()


def _reminders_body(
    reminders: Iterable[dict[str, Any] | None] | None,
    use_default_reminders: bool | None,
) -> dict[str, Any] | None:
    """Calendar-API reminders object, or None when the caller said nothing.

    override semantics match Google's: useDefault=false with overrides replaces
    every reminder; useDefault=true restores the calendar defaults.
    """
    overrides: list[dict[str, Any]] = []
    for item in reminders or ():
        if not isinstance(item, dict):
            continue
        try:
            minutes = int(item.get("minutes"))
        except (TypeError, ValueError):
            continue
        overrides.append({"method": str(item.get("method") or "popup"), "minutes": minutes})
    if overrides or use_default_reminders is not None:
        return {
            "useDefault": bool(use_default_reminders) if use_default_reminders is not None else False,
            "overrides": overrides,
        }
    return None


def _event_time(value: str, timezone: str | None) -> dict[str, str]:
    text = str(value or "").strip()
    if _DATE_ONLY.match(text):
        return {"date": text}
    # a naive dateTime ("2026-11-26T15:00:00") means *local time* to the person
    # speaking — Google's API demands an offset, so attach the server's local one
    text = _as_rfc3339(text) or text
    out = {"dateTime": text}
    if timezone:
        out["timeZone"] = str(timezone)
    return out


def _event_shape(event: dict[str, Any]) -> dict[str, Any]:
    start = event.get("start") or {}
    end = event.get("end") or {}
    all_day = bool(start.get("date")) and not start.get("dateTime")
    return {
        "id": str(event.get("id") or ""),
        "summary": str(event.get("summary") or ""),
        "start": str(start.get("dateTime") or start.get("date") or ""),
        "end": str(end.get("dateTime") or end.get("date") or ""),
        "all_day": all_day,
        "location": str(event.get("location") or ""),
        "description": str(event.get("description") or ""),
        "reminders": (event.get("reminders") or {}).get("overrides") or [],
        "attendees": [
            {
                "email": str(attendee.get("email") or ""),
                "displayName": str(attendee.get("displayName") or ""),
                "responseStatus": str(attendee.get("responseStatus") or ""),
            }
            for attendee in event.get("attendees") or []
        ],
        "html_link": str(event.get("htmlLink") or ""),
        "status": str(event.get("status") or ""),
    }
