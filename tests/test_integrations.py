"""Tests for the Google + Slack integration clients.

No network: GoogleClient._request / SlackClient._call are monkeypatched with
fakes, and the two HTTP-seam tests use httpx.MockTransport.
"""
from __future__ import annotations

import base64
from email import policy
from email.parser import BytesParser
from typing import Any, Callable

import httpx
import pytest

from otto.integrations import IntegrationError, NotConfigured
from otto.integrations.google import GoogleClient, build_credentials
from otto.integrations.slack import SlackClient
from otto.settings import Settings

MESSAGE_HEADERS = [
    {"name": "From", "value": "Sarah Chen <sarah@example.com>"},
    {"name": "To", "value": "me@example.com"},
    {"name": "Subject", "value": "Invoice 42"},
    {"name": "Date", "value": "Mon, 4 Mar 2026 09:15:00 +0100"},
]


# --------------------------------------------------------------------------- helpers
def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _patch_request(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[str, str, dict | None, dict | None], dict],
) -> list[dict[str, Any]]:
    """Replace the single Google HTTP seam and record the calls it received."""
    calls: list[dict[str, Any]] = []

    async def fake_request(self, method, url, *, params=None, json=None):
        calls.append({"method": method, "url": url, "params": params, "json": json})
        return handler(method, url, params, json)

    monkeypatch.setattr(GoogleClient, "_request", fake_request)
    return calls


def _patch_call(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[str, dict[str, Any]], dict],
) -> list[dict[str, Any]]:
    """Replace the single Slack HTTP seam and record the calls it received."""
    calls: list[dict[str, Any]] = []

    async def fake_call(self, method, **payload):
        calls.append({"method": method, "payload": payload})
        return handler(method, payload)

    monkeypatch.setattr(SlackClient, "_call", fake_call)
    return calls


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        root=tmp_path,
        google_client_secrets="secrets/google_client.json",
        google_token_path="secrets/google_token.json",
        gcal_calendar_id="primary",
        slack_bot_token="xoxb-test",
    )


@pytest.fixture
def google(settings) -> GoogleClient:
    return GoogleClient(settings)


@pytest.fixture
def slack(settings) -> SlackClient:
    return SlackClient(settings)


# ----------------------------------------------------------------------- google status
def test_google_status_reports_config_and_token(tmp_path, settings):
    client = GoogleClient(settings)
    before = client.status()
    assert before["configured"] is False
    assert before["authorized"] is False
    assert before["scopes"]
    assert "Connections" in before["detail"]
    assert client.connected is False

    settings.google_token_file.parent.mkdir(parents=True, exist_ok=True)
    settings.google_token_file.write_text('{"refresh_token": "r", "client_id": "i", "client_secret": "s"}')
    after = client.status()
    assert after["connected"] is True
    assert after["authorized"] is True
    assert client.connected is True


def test_google_missing_token_is_not_configured(settings):
    assert GoogleClient(settings).status()["authorized"] is False
    with pytest.raises(NotConfigured) as excinfo:
        build_credentials(settings)
    assert "google_auth.py" in str(excinfo.value)


async def test_ensure_token_refreshes_and_persists_token_file(monkeypatch, google):
    class FakeCredentials:
        valid = False
        token = "expired-token"
        refresh_token = "refresh-token"
        refreshed = 0

        def refresh(self, request):
            FakeCredentials.refreshed += 1
            self.token = "new-token"
            self.valid = True

        def to_json(self) -> str:
            return '{"token": "new-token", "refresh_token": "refresh-token"}'

    google.store.save('{"token": "expired-token", "refresh_token": "refresh-token"}')
    monkeypatch.setattr(
        "otto.integrations.google.Credentials.from_authorized_user_info",
        lambda info, scopes=None: FakeCredentials(),
    )
    assert await google._ensure_token() == "new-token"
    assert FakeCredentials.refreshed == 1
    assert google.store.load() == '{"token": "new-token", "refresh_token": "refresh-token"}'
    assert google.store.path.stat().st_mode & 0o777 == 0o600

    # a valid cached token must not trigger another refresh
    assert await google._ensure_token() == "new-token"
    assert FakeCredentials.refreshed == 1


# --------------------------------------------------------------------------- gmail
async def test_gmail_search_maps_metadata_and_unread(monkeypatch, google):
    def handler(method, url, params, payload):
        if url.endswith("/messages"):
            return {"messages": [{"id": "m1", "threadId": "t1"}], "resultSizeEstimate": 1}
        if url.endswith("/messages/m1"):
            return {
                "id": "m1",
                "threadId": "t1",
                "labelIds": ["INBOX", "UNREAD"],
                "snippet": "Your invoice is attached",
                "internalDate": "1772613900000",
                "payload": {"headers": MESSAGE_HEADERS},
            }
        raise AssertionError(f"unexpected url {url}")

    calls = _patch_request(monkeypatch, handler)
    items = await google.gmail_search("has:attachment", max_results=5)

    assert len(items) == 1
    message = items[0]
    assert message == {
        "id": "m1",
        "thread_id": "t1",
        "from_": "Sarah Chen <sarah@example.com>",
        "to": "me@example.com",
        "subject": "Invoice 42",
        "snippet": "Your invoice is attached",
        "date": "Mon, 4 Mar 2026 09:15:00 +0100",
        "unread": True,
        "labels": ["INBOX", "UNREAD"],
    }
    assert calls[0]["method"] == "GET"
    assert calls[0]["params"]["q"] == "has:attachment"
    assert calls[0]["params"]["maxResults"] == 5
    assert calls[0]["params"]["includeSpamTrash"] == "false"
    assert calls[1]["params"] == {
        "format": "metadata",
        "metadataHeaders": ["From", "To", "Subject", "Date"],
    }


async def test_gmail_search_marks_read_when_not_unread(monkeypatch, google):
    def handler(method, url, params, payload):
        if url.endswith("/messages"):
            return {"messages": [{"id": "m2"}]}
        return {
            "id": "m2",
            "threadId": "t2",
            "labelIds": ["INBOX"],
            "snippet": "seen",
            "payload": {"headers": MESSAGE_HEADERS},
        }

    _patch_request(monkeypatch, handler)
    items = await google.gmail_search("")
    assert items[0]["unread"] is False


async def test_gmail_get_extracts_plain_text_and_attachments(monkeypatch, google):
    payload = {
        "mimeType": "multipart/mixed",
        "headers": MESSAGE_HEADERS + [{"name": "Cc", "value": "cfo@example.com"}],
        "parts": [
            {
                "mimeType": "multipart/alternative",
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": _b64("Plain body wins")}},
                    {"mimeType": "text/html", "body": {"data": _b64("<p>HTML body</p>")}},
                ],
            },
            {
                "mimeType": "application/pdf",
                "filename": "invoice-42.pdf",
                "body": {"size": 4096, "attachmentId": "att1"},
            },
        ],
    }

    def handler(method, url, params, json):
        assert url.endswith("/messages/m1")
        assert params == {"format": "full"}
        return {
            "id": "m1",
            "threadId": "t1",
            "labelIds": ["UNREAD"],
            "snippet": "see attached",
            "payload": payload,
        }

    _patch_request(monkeypatch, handler)
    message = await google.gmail_get("m1")

    assert message["body_text"] == "Plain body wins"
    assert message["cc"] == "cfo@example.com"
    assert message["unread"] is True
    assert message["attachments"] == [
        {"filename": "invoice-42.pdf", "size": 4096, "mime": "application/pdf"}
    ]


async def test_gmail_get_falls_back_to_stripped_html(monkeypatch, google):
    payload = {
        "mimeType": "text/html",
        "headers": MESSAGE_HEADERS,
        "body": {"data": _b64("<html><body><p>Hello <b>there</b></p><br/>Bye</body></html>")},
    }

    def handler(method, url, params, json):
        return {"id": "m3", "threadId": "t3", "labelIds": [], "payload": payload}

    _patch_request(monkeypatch, handler)
    message = await google.gmail_get("m3")
    assert "Hello there" in message["body_text"]
    assert "<" not in message["body_text"]
    assert "Bye" in message["body_text"]


async def test_gmail_send_builds_rfc822_message(monkeypatch, google):
    def handler(method, url, params, payload):
        assert method == "POST"
        assert url.endswith("/messages/send")
        raw = payload["raw"]
        parsed = BytesParser(policy=policy.default).parsebytes(_unb64(raw))
        assert parsed["To"] == "sarah@example.com, tom@example.com"
        assert parsed["Cc"] == "cfo@example.com"
        assert parsed["Subject"] == "Q3 numbers"
        assert "Here are the numbers." in parsed.get_content()
        return {"id": "sent1", "threadId": "thr1"}

    _patch_request(monkeypatch, handler)
    result = await google.gmail_send(
        to=["sarah@example.com", "tom@example.com"],
        subject="Q3 numbers",
        body="Here are the numbers.",
        cc=["cfo@example.com"],
    )
    assert result == {"id": "sent1", "thread_id": "thr1"}


async def test_gmail_draft_posts_raw_to_drafts(monkeypatch, google):
    def handler(method, url, params, payload):
        assert method == "POST"
        assert url.endswith("/drafts")
        parsed = BytesParser(policy=policy.default).parsebytes(_unb64(payload["message"]["raw"]))
        assert parsed["Subject"] == "Draft subject"
        return {"id": "draft1", "message": {"id": "msg1", "threadId": "thr1"}}

    _patch_request(monkeypatch, handler)
    result = await google.gmail_draft(to="sarah@example.com", subject="Draft subject", body="hi")
    assert result == {"id": "draft1", "message_id": "msg1"}


# ------------------------------------------------------------------------ calendar
async def test_calendar_create_datetime_body_and_shape(monkeypatch, google):
    def handler(method, url, params, payload):
        assert method == "POST"
        assert url == "https://www.googleapis.com/calendar/v3/calendars/primary/events"
        assert payload == {
            "summary": "Design review",
            "start": {"dateTime": "2026-03-04T15:00:00+01:00", "timeZone": "Europe/Berlin"},
            "end": {"dateTime": "2026-03-04T16:00:00+01:00", "timeZone": "Europe/Berlin"},
            "location": "Room 2",
            "attendees": [{"email": "sarah@example.com"}],
        }
        return {
            "id": "evt1",
            "summary": "Design review",
            "status": "confirmed",
            "htmlLink": "https://calendar.google.com/event?eid=evt1",
            "location": "Room 2",
            "start": {"dateTime": "2026-03-04T15:00:00+01:00"},
            "end": {"dateTime": "2026-03-04T16:00:00+01:00"},
            "attendees": [
                {"email": "sarah@example.com", "displayName": "Sarah", "responseStatus": "accepted"}
            ],
        }

    _patch_request(monkeypatch, handler)
    event = await google.calendar_create(
        summary="Design review",
        start="2026-03-04T15:00:00+01:00",
        end="2026-03-04T16:00:00+01:00",
        location="Room 2",
        attendees=["sarah@example.com"],
        timezone="Europe/Berlin",
    )
    assert event == {
        "id": "evt1",
        "summary": "Design review",
        "start": "2026-03-04T15:00:00+01:00",
        "end": "2026-03-04T16:00:00+01:00",
        "all_day": False,
        "location": "Room 2",
        "description": "",
        "reminders": [],
        "attendees": [
            {"email": "sarah@example.com", "displayName": "Sarah", "responseStatus": "accepted"}
        ],
        "html_link": "https://calendar.google.com/event?eid=evt1",
        "status": "confirmed",
    }


async def test_calendar_create_all_day_uses_date_fields(monkeypatch, google):
    def handler(method, url, params, payload):
        assert payload["start"] == {"date": "2026-03-04"}
        assert payload["end"] == {"date": "2026-03-05"}
        return {
            "id": "evt2",
            "summary": "Offsite",
            "start": {"date": "2026-03-04"},
            "end": {"date": "2026-03-05"},
        }

    _patch_request(monkeypatch, handler)
    event = await google.calendar_create("Offsite", "2026-03-04", "2026-03-05")
    assert event["all_day"] is True
    assert event["start"] == "2026-03-04"
    assert event["end"] == "2026-03-05"


async def test_calendar_list_requests_ordered_single_events(monkeypatch, google):
    def handler(method, url, params, payload):
        assert method == "GET"
        assert params["singleEvents"] == "true"
        assert params["orderBy"] == "startTime"
        assert params["timeMin"] == "2026-03-04T00:00:00+01:00"
        assert params["maxResults"] == 3
        return {"items": [{"id": "e1", "start": {"dateTime": "2026-03-04T09:00:00+01:00"}}]}

    _patch_request(monkeypatch, handler)
    events = await google.calendar_list(
        time_min="2026-03-04T00:00:00+01:00", time_max="2026-03-05T00:00:00+01:00", max_results=3
    )
    assert events[0]["id"] == "e1"


async def test_calendar_delete_returns_id_only(monkeypatch, google):
    def handler(method, url, params, payload):
        assert method == "DELETE"
        assert url.endswith("/events/evt1")
        return {}  # Google answers 204 No Content

    _patch_request(monkeypatch, handler)
    assert await google.calendar_delete("evt1") == {"id": "evt1", "deleted": True}


# ------------------------------------------------------------- google http seam
async def test_request_retries_once_after_refreshing_on_401(monkeypatch, settings):
    requests: list[httpx.Request] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(401, json={"error": {"message": "Invalid Credentials"}})
        return httpx.Response(200, json={"messages": [{"id": "m1"}]})

    async def fake_ensure_token(self, *, force=False):
        return "fresh-token" if force else "stale-token"

    monkeypatch.setattr(GoogleClient, "_ensure_token", fake_ensure_token)
    client = GoogleClient(settings)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(transport_handler))
    try:
        data = await client._request(
            "GET", "https://gmail.googleapis.com/gmail/v1/users/me/messages", params={"q": "x"}
        )
    finally:
        await client.aclose()

    assert data == {"messages": [{"id": "m1"}]}
    assert [r.headers["authorization"] for r in requests] == [
        "Bearer stale-token",
        "Bearer fresh-token",
    ]


async def test_request_raises_after_a_single_retry(monkeypatch, settings):
    requests: list[httpx.Request] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(401, json={"error": {"message": "Invalid Credentials"}})

    async def fake_ensure_token(self, *, force=False):
        return "token"

    monkeypatch.setattr(GoogleClient, "_ensure_token", fake_ensure_token)
    client = GoogleClient(settings)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(transport_handler))
    try:
        with pytest.raises(IntegrationError) as excinfo:
            await client._request("GET", "https://gmail.googleapis.com/gmail/v1/users/me/labels")
    finally:
        await client.aclose()

    assert len(requests) == 2
    assert "401" in str(excinfo.value)


async def test_request_error_includes_status_and_body_excerpt(monkeypatch, settings):
    def transport_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "Insufficient Permission"}})

    async def fake_ensure_token(self, *, force=False):
        return "token"

    monkeypatch.setattr(GoogleClient, "_ensure_token", fake_ensure_token)
    client = GoogleClient(settings)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(transport_handler))
    try:
        with pytest.raises(IntegrationError) as excinfo:
            await client._request("GET", "https://gmail.googleapis.com/gmail/v1/users/me/labels")
    finally:
        await client.aclose()

    message = str(excinfo.value)
    assert "403" in message
    assert "Insufficient Permission" in message


# --------------------------------------------------------------------------- slack
async def test_slack_status_reports_token(settings):
    assert SlackClient(settings).status()["configured"] is True
    settings.slack_bot_token = ""
    status = SlackClient(settings).status()
    assert status["configured"] is False
    assert "Connections" in status["detail"]
    # a per-user token overrides whatever is in the environment
    assert SlackClient(settings, token="xoxb-user").status()["configured"] is True


async def test_slack_call_requires_a_token(settings):
    settings.slack_bot_token = ""
    with pytest.raises(NotConfigured):
        await SlackClient(settings).channels()


async def test_slack_resolve_channel_by_name(monkeypatch, slack):
    def handler(method, payload):
        assert method == "conversations.list"
        return {
            "ok": True,
            "channels": [
                {"id": "C0GENERAL", "name": "general", "num_members": 12, "topic": {"value": "hi"}},
                {"id": "C0DESIGN", "name": "design", "is_private": True},
            ],
        }

    calls = _patch_call(monkeypatch, handler)
    assert await slack.resolve_channel("general") == "C0GENERAL"
    assert await slack.resolve_channel("#General") == "C0GENERAL"
    assert await slack.resolve_channel("C0ABCDEF1") == "C0ABCDEF1"
    assert calls[-1]["method"] == "conversations.list"

    with pytest.raises(IntegrationError) as excinfo:
        await slack.resolve_channel("gerenal")
    assert "#general" in str(excinfo.value)


async def test_slack_history_resolves_user_names_oldest_first(monkeypatch, slack):
    def handler(method, payload):
        if method == "conversations.history":
            assert payload["channel"] == "C0GENERAL"
            assert payload["limit"] == 5
            return {
                "ok": True,
                "messages": [  # Slack returns newest first
                    {"ts": "1700000002.000200", "user": "U2", "text": "newer", "thread_ts": "1700000002.000200", "replies": [{"user": "U1"}, {"user": "U3"}]},
                    {"ts": "1700000001.000100", "subtype": "bot_message", "text": "   "},
                    {"ts": "1700000000.000100", "user": "U1", "text": "older"},
                ],
            }
        if method == "users.list":
            return {
                "ok": True,
                "members": [
                    {"id": "U1", "name": "sarah", "profile": {"real_name": "Sarah Chen"}},
                    {"id": "U2", "name": "tom", "profile": {"real_name": "Tom Ruiz"}},
                ],
            }
        raise AssertionError(f"unexpected method {method}")

    calls = _patch_call(monkeypatch, handler)
    messages = await slack.history("C0GENERAL", limit=5)

    assert [m["ts"] for m in messages] == ["1700000000.000100", "1700000002.000200"]
    assert messages[0]["user_name"] == "Sarah Chen"
    assert messages[1]["user_name"] == "Tom Ruiz"
    assert messages[1]["user"] == "U2"
    assert messages[1]["replies"] == 2
    assert messages[1]["thread_ts"] == "1700000002.000200"
    assert messages[0]["permalink"] == "https://slack.com/archives/C0GENERAL/p1700000000000100"
    # users.list is cached: a second history call must not re-fetch it
    await slack.history("C0GENERAL", limit=5)
    assert [c["method"] for c in calls].count("users.list") == 1


async def test_slack_search_flattens_matches(monkeypatch, slack):
    def handler(method, payload):
        if method == "search.messages":
            assert payload["query"] == "invoice in:#design"
            assert payload["count"] == 3
            return {
                "ok": True,
                "messages": {
                    "matches": [
                        {
                            "ts": "1700000001.000100",
                            "user": "U1",
                            "text": "invoice attached",
                            "permalink": "https://acme.slack.com/archives/C0DESIGN/p1700000001000100",
                            "channel": {"id": "C0DESIGN", "name": "design"},
                        }
                    ]
                },
            }
        if method == "users.list":
            return {"ok": True, "members": [{"id": "U1", "name": "sarah", "profile": {"real_name": "Sarah Chen"}}]}
        raise AssertionError(f"unexpected method {method}")

    _patch_call(monkeypatch, handler)
    results = await slack.search("invoice in:#design", count=3)
    assert results == [
        {
            "ts": "1700000001.000100",
            "channel": "C0DESIGN",
            "channel_name": "design",
            "user": "U1",
            "user_name": "Sarah Chen",
            "text": "invoice attached",
            "permalink": "https://acme.slack.com/archives/C0DESIGN/p1700000001000100",
        }
    ]


async def test_slack_post_message_form_encodes_payload(settings):
    requests: list[httpx.Request] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "ts": "1700000009.000900", "channel": "C0GENERAL"})

    slack = SlackClient(settings)
    slack._http = httpx.AsyncClient(transport=httpx.MockTransport(transport_handler))
    try:
        result = await slack.post_message("C0GENERAL", "hello there", thread_ts="1700000001.000100")
    finally:
        await slack.aclose()

    assert result == {"ok": True, "ts": "1700000009.000900", "channel": "C0GENERAL"}
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://slack.com/api/chat.postMessage"
    assert request.headers["authorization"] == "Bearer xoxb-test"
    body = request.content.decode()
    assert "channel=C0GENERAL" in body
    assert "text=hello+there" in body
    assert "thread_ts=1700000001.000100" in body
    assert "unfurl_links=false" in body


async def test_slack_call_maps_ok_false_to_integration_error(settings):
    def transport_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"ok": False, "error": "missing_scope", "needed": "channels:history"}
        )

    slack = SlackClient(settings)
    slack._http = httpx.AsyncClient(transport=httpx.MockTransport(transport_handler))
    try:
        with pytest.raises(IntegrationError) as excinfo:
            await slack.history("C0GENERAL")
    finally:
        await slack.aclose()

    message = str(excinfo.value)
    assert "missing_scope" in message
    assert "channels:history" in message


async def test_people_profile_and_search(monkeypatch, settings):
    import httpx

    from otto.integrations.google import GoogleClient

    def transport_handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != "people.googleapis.com":
            return httpx.Response(500, json={"error": "wrong host"})
        if request.url.path.endswith("/people/me"):
            return httpx.Response(200, json={
                "names": [{"displayName": "Guy Sandler"}],
                "emailAddresses": [{"value": "guy@sandler.co.il"}],
            })
        assert request.url.params["query"] == "david"
        return httpx.Response(200, json={
            "connections": [
                {"names": [{"displayName": "David Byrne"}],
                 "emailAddresses": [{"value": "david@boss.com"}, {"value": "david2@boss.com"}]},
                {"names": [{"displayName": "Dana"}], "emailAddresses": []},
            ]
        })

    async def fake_ensure_token(self, *, force=False):
        return "token"

    monkeypatch.setattr(GoogleClient, "_ensure_token", fake_ensure_token)
    client = GoogleClient(settings)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(transport_handler))
    try:
        profile = await client.people_profile()
        assert profile == {"name": "Guy Sandler", "email": "guy@sandler.co.il"}

        matches = await client.people_search("david")
        assert matches[0] == {"name": "David Byrne", "email": "david@boss.com",
                              "emails": ["david@boss.com", "david2@boss.com"]}
        assert matches[1]["name"] == "Dana"
    finally:
        await client.aclose()
