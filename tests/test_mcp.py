"""MCP client, tool policy and manager (against an in-process fake server)."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from otto.mcp.client import MCPClient, MCPError
from otto.mcp.manager import MCPManager
from otto.mcp.policy import confirmation_for, policy_for
from otto.settings import Settings
from otto.tools.base import ToolContext

from .fake_mcp import FakeMCPServer, calendar_tools, gmail_tools


def settings_for(tmp_path, *, mode="mcp", fallback=True, servers=("gmail",)) -> Settings:
    app_cfg = {
        "mcp": {
            "enabled": True,
            "mode": mode,
            "allow_rest_fallback": fallback,
            "timeout_s": 5,
            "tool_cache_s": 0,
            "discover_timeout_s": 3,
            "servers": {
                name: {"url": f"https://{name}mcp.test/mcp/v1"} for name in servers
            },
        }
    }
    return Settings(root=tmp_path, app_cfg=app_cfg)


def manager_for(tmp_path, server: FakeMCPServer, *, token="tok", name="gmail", **kwargs) -> MCPManager:
    settings = settings_for(tmp_path, **kwargs)

    async def provider() -> str:
        return token

    return MCPManager(settings, token_provider=provider, transports={name: httpx.ASGITransport(app=server.app)})


# --------------------------------------------------------------------- client
async def test_client_handshake_and_tool_listing():
    server = FakeMCPServer(tools=gmail_tools(), require_token="tok")
    client = MCPClient(
        "https://gmailmcp.test/mcp/v1",
        token_provider=lambda: _value("tok"),
        transport=httpx.ASGITransport(app=server.app),
    )
    info = await client.initialize()
    assert info["serverInfo"]["name"] == "fake-workspace"
    assert client._session_id == "sess-1234"

    tools = await client.list_tools()
    assert [tool.name for tool in tools] == ["search_threads", "create_draft", "get_message"]
    assert tools[0].read_only is True
    assert tools[1].destructive is False
    # protocol version header must be sent once initialized
    assert server.protocol_versions and server.protocol_versions[-1]
    await client.aclose()


async def test_client_parses_sse_responses():
    server = FakeMCPServer(tools=gmail_tools(), require_token="tok", sse=True)
    client = MCPClient(
        "https://gmailmcp.test/mcp/v1",
        token_provider=lambda: _value("tok"),
        transport=httpx.ASGITransport(app=server.app),
    )
    tools = await client.list_tools()
    assert len(tools) == 3
    await client.aclose()


async def test_client_reinitializes_after_session_expiry():
    server = FakeMCPServer(tools=gmail_tools(), require_token="tok")
    client = MCPClient(
        "https://gmailmcp.test/mcp/v1",
        token_provider=lambda: _value("tok"),
        transport=httpx.ASGITransport(app=server.app),
    )
    await client.list_tools()
    server.session_expired = True
    tools = await client.list_tools()  # must transparently re-handshake
    assert len(tools) == 3
    assert server.initialized == 2
    await client.aclose()


async def test_client_explains_a_non_agentic_client():
    server = FakeMCPServer(
        tools=gmail_tools(),
        require_token="tok",
        forbid=True,
        forbidden_detail='Client is not configured as an "AI-powered agent" (is_agentic=false)',
    )
    client = MCPClient(
        "https://gmailmcp.test/mcp/v1",
        token_provider=lambda: _value("tok"),
        transport=httpx.ASGITransport(app=server.app),
    )
    with pytest.raises(MCPError) as excinfo:
        await client.list_tools()
    assert excinfo.value.reason == "forbidden"
    assert "AI-powered agent" in str(excinfo.value)
    await client.aclose()


async def test_client_reports_rpc_errors():
    server = FakeMCPServer(tools=gmail_tools(), results={})
    client = MCPClient(
        "https://gmailmcp.test/mcp/v1",
        token_provider=lambda: _value("tok"),
        transport=httpx.ASGITransport(app=server.app),
    )
    with pytest.raises(MCPError) as excinfo:
        await client.call_tool("nope", {})
    assert "unknown tool" in str(excinfo.value)
    await client.aclose()


# --------------------------------------------------------------------- policy
def test_policy_maps_names_and_confirmations():
    search = policy_for("gmail", "search_threads")
    assert search.alias == "gmail_search"
    draft = policy_for("gmail", "create_draft")
    assert draft.alias == "gmail_draft"
    create = policy_for("calendar", "create_event")
    assert create.alias == "calendar_create_event"
    assert create.requires_confirmation is True
    delete = policy_for("calendar", "delete_event")
    assert delete.danger is True
    unknown = policy_for("notion", "append_block")
    assert unknown.alias == "notion_append_block"


def test_every_google_workspace_tool_has_a_policy():
    """Guards against Google adding tools we would silently expose as raw names."""
    gmail = [
        "create_draft", "list_drafts", "get_draft", "get_thread", "get_message", "search_threads",
        "label_thread", "unlabel_thread", "apply_sensitive_thread_label", "trash_thread",
        "untrash_thread", "mark_thread_spam", "unmark_thread_spam", "list_labels", "label_message",
        "update_message_labels", "unlabel_message", "apply_sensitive_message_label", "trash_message",
        "untrash_message", "mark_message_spam", "unmark_message_spam", "create_label",
    ]
    calendar = [
        "list_events", "get_event", "list_calendars", "suggest_time", "create_event",
        "update_event", "delete_event", "respond_to_event", "search_events",
    ]
    from otto.mcp.policy import POLICIES

    assert [t for t in gmail if ("gmail", t) not in POLICIES] == []
    assert [t for t in calendar if ("calendar", t) not in POLICIES] == []
    # the ones that change or destroy data must ask first
    for name in ("trash_thread", "trash_message", "mark_message_spam", "apply_sensitive_message_label"):
        assert POLICIES[("gmail", name)].requires_confirmation is True, name
    # reads and reversible edits must not nag (None = follow Google's read-only hint)
    for name in ("search_threads", "get_message", "label_message", "unlabel_message", "untrash_message"):
        assert POLICIES[("gmail", name)].requires_confirmation is not True, name


def test_prune_schema_drops_form_only_properties():
    from otto.mcp.manager import prune_schema

    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Title of the event " * 40},
            "attachments": {"type": "array", "items": {"type": "object"}},
            "conferenceData": {"type": "object", "properties": {"x": {"type": "string"}}},
            "start": {"type": "object", "properties": {"dateTime": {"type": "string", "description": "when " * 60}}},
            "pageToken": {"type": "string"},
            # reminders are spoken requests — they must survive pruning
            "overrideReminders": {"type": "array", "items": {"$ref": "#/$defs/Reminder"}},
            "useDefaultReminders": {"type": "boolean", "x-google-enum-descriptions": ["bloat"] * 8},
        },
        "required": ["summary", "attachments"],
    }
    pruned = prune_schema(schema, max_description=60)
    props = pruned["properties"]
    assert "attachments" not in props and "conferenceData" not in props and "pageToken" not in props
    assert "overrideReminders" in props and "useDefaultReminders" in props
    assert "x-google-enum-descriptions" not in props["useDefaultReminders"]
    assert props["summary"]["description"].endswith("…") and len(props["summary"]["description"]) <= 60
    assert "dateTime" in props["start"]["properties"]
    # a required field we removed must not stay required
    assert pruned["required"] == ["summary"]


def test_confirmation_follows_annotations_for_unknown_tools():
    class Info:
        destructive = True

    assert confirmation_for(policy_for("x", "y"), Info()) is True
    assert confirmation_for(policy_for("x", "y"), None) is False


# -------------------------------------------------------------------- manager
async def test_manager_builds_tools_and_calls_them(tmp_path):
    server = FakeMCPServer(
        tools=gmail_tools(),
        require_token="tok",
        results={
            "search_threads": {
                "threads": [
                    {"id": "t1", "subject": "Lease renewal", "from": "sarah@acme.com", "snippet": "can we push it?", "unread": True},
                    {"id": "t2", "subject": "Invoice", "from": "billing@acme.com", "snippet": "paid"},
                ]
            },
            "create_draft": {"id": "d1", "subject": "Re: Lease", "toRecipients": ["sarah@acme.com"]},
        },
    )
    manager = manager_for(tmp_path, server)
    tools = await manager.discover()
    names = {tool.name for tool in tools}
    assert names == {"gmail_search", "gmail_draft", "gmail_read"}
    assert manager.status()["servers"][0]["ok"] is True

    search = manager.get("gmail_search")
    ctx = ToolContext(settings=manager.settings)
    result = await search.invoke(ctx, {"query": "from:sarah"})
    assert result.ok
    assert "Lease renewal" in result.summary
    assert result.display["kind"] == "email_list"
    assert result.display["items"][0]["id"] == "t1"
    assert result.display["items"][0]["unread"] is True
    # empty arguments are not forwarded
    assert server.calls[0][0] == "search_threads"

    draft = manager.get("gmail_draft")
    assert draft.needs_confirmation({}) is False
    created = await draft.invoke(ctx, {"to": ["sarah@acme.com"], "subject": "Re: Lease", "body": "Tuesday works."})
    assert created.ok
    assert created.display["kind"] == "email_draft"
    assert created.display["draftId"] == "d1"
    await manager.aclose()


async def test_manager_marks_destructive_tools_for_confirmation(tmp_path):
    server = FakeMCPServer(
        tools=calendar_tools(),
        require_token="tok",
        results={
            "delete_event": {"event": {"id": "e1", "summary": "Dentist", "start": {"dateTime": "2026-11-26T15:00:00Z"}}},
            "list_events": {
                "summary": "primary",
                "events": [
                    {"id": "e1", "summary": "Dentist", "start": {"dateTime": "2026-11-26T15:00:00Z"}, "location": "Clinic"},
                    {"id": "e2", "summary": "Design sync", "start": {"dateTime": "2026-11-25T10:00:00-08:00"}},
                ],
            },
            "create_event": {"event": {"id": "e3", "summary": "Design sync", "start": {"dateTime": "2026-11-25T16:00:00-08:00"}}},
        },
    )
    manager = manager_for(tmp_path, server, name="calendar", servers=("calendar",))
    await manager.discover()
    delete = manager.get("calendar_delete_event")
    assert delete.needs_confirmation({"eventId": "e1"}) is True
    assert delete.danger is True
    assert "delete" in delete.confirmation_text({"eventId": "e1"}).lower()

    create = manager.get("calendar_create_event")
    assert create.needs_confirmation({"summary": "Design sync"}) is True
    assert "Design sync" in create.confirmation_text({"summary": "Design sync", "start": {"dateTime": "2026-11-25T16:00:00-08:00"}})

    listing = manager.get("calendar_list_events")
    assert listing.needs_confirmation({}) is False
    result = await listing.invoke(ToolContext(settings=manager.settings), {"startTime": "2026-11-25T00:00:00Z"})
    assert result.ok, result.error
    assert result.display["kind"] == "event_list"
    assert [item["id"] for item in result.display["items"]] == ["e1", "e2"]
    assert "Design sync" in result.summary
    await manager.aclose()


async def test_manager_survives_a_broken_server(tmp_path):
    server = FakeMCPServer(
        tools=gmail_tools(),
        require_token="tok",
        forbid=True,
        forbidden_detail="gmailmcp.googleapis.com has not been used in project 123",
    )
    manager = manager_for(tmp_path, server)
    tools = await manager.discover()
    assert tools == []
    status = manager.status()
    assert status["active"] is False
    assert "MCP service is not enabled" in status["servers"][0]["detail"]
    assert manager.drain_notices()  # the user gets told once
    await manager.aclose()


async def test_manager_without_a_token_lists_but_does_not_offer_tools(tmp_path):
    """Discovery works anonymously; use is gated on the user's token."""
    manager = manager_for(tmp_path, FakeMCPServer(tools=gmail_tools()), token="")

    async def failing() -> str:
        raise RuntimeError("Google is not connected")

    manager.token_provider = failing
    tools = await manager.discover()
    assert {tool.name for tool in tools} == {"gmail_search", "gmail_draft", "gmail_read"}
    assert manager.schemas() == []          # not offered to the model
    status = manager.status()
    assert status["active"] is False
    assert status["tokenReady"] is False
    assert "not connected" in status["tokenError"].lower()
    # the catalog still shows them, greyed out, so the UI can explain the setup
    assert all(entry["configured"] is False for entry in manager.catalog())


async def test_manager_prompt_note_explains_missing_send(tmp_path):
    server = FakeMCPServer(tools=gmail_tools(), require_token="tok")
    manager = manager_for(tmp_path, server)
    await manager.discover()
    note = manager.prompt_note()
    assert "cannot send" in note
    await manager.aclose()


async def test_manager_mode_rest_ignores_mcp(tmp_path):
    server = FakeMCPServer(tools=gmail_tools(), require_token="tok")
    manager = manager_for(tmp_path, server, mode="rest")
    assert await manager.discover() == []
    assert manager.uses_mcp is False
    assert manager.authoritative is False


async def test_self_test_reports_missing_token(tmp_path):
    manager = manager_for(tmp_path, FakeMCPServer(tools=gmail_tools()), token="")

    async def failing() -> str:
        raise RuntimeError("Google is not connected")

    manager.token_provider = failing
    result = await manager.self_test()
    assert result["ok"] is False
    assert result["tokenReady"] is False
    assert result["servers"]["gmail"]["reason"] == "unauthorized"


async def test_self_test_calls_a_read_tool_per_server(tmp_path):
    server = FakeMCPServer(
        tools=gmail_tools() + [{
            "name": "list_labels",
            "title": "List labels",
            "description": "List Gmail labels.",
            "inputSchema": {"type": "object", "properties": {}},
            "annotations": {"readOnlyHint": True},
        }],
        require_token="tok",
        results={"list_labels": {"labels": [{"name": "INBOX"}, {"name": "Receipts"}]}},
    )
    manager = manager_for(tmp_path, server)
    result = await manager.self_test()
    assert result["ok"] is True
    assert result["servers"]["gmail"]["ok"] is True
    assert result["servers"]["gmail"]["tool"] == "list_labels"
    assert "INBOX" in result["servers"]["gmail"]["detail"]
    assert server.calls[-1][0] == "list_labels"
    await manager.aclose()


async def test_self_test_explains_a_forbidden_call(tmp_path):
    server = FakeMCPServer(tools=gmail_tools(), require_token="tok", forbid=True,
                           forbidden_detail='Client is not configured as an "AI-powered agent"')
    manager = manager_for(tmp_path, server)
    result = await manager.self_test()
    assert result["ok"] is False
    assert result["servers"]["gmail"]["reason"] == "forbidden"
    assert "AI-powered agent" in result["servers"]["gmail"]["detail"]
    await manager.aclose()


# ------------------------------------------------- argument normalisation
def test_normalise_arguments_folds_rest_names_onto_mcp_names():
    from otto.mcp.manager import normalise_arguments

    schema = {"properties": {"startTime": {}, "endTime": {}, "pageSize": {}, "calendarId": {}}}
    got = normalise_arguments(
        {
            "time_min": "2026-11-30T00:00:00-08:00",
            "time_max": "2026-12-07T00:00:00-08:00",
            "max_results": "25",
            "bogus": True,
        },
        schema,
    )
    assert got["startTime"].startswith("2026-11-30")
    assert got["endTime"].startswith("2026-12-07")
    assert got["pageSize"] == 25            # numeric string coerced
    assert got["__dropped__"] == ["bogus"]  # never sent to the server
    assert "time_min" not in got


def test_normalise_arguments_handles_ids_and_camel_case():
    from otto.mcp.manager import normalise_arguments

    schema = {"properties": {"eventId": {}, "messageId": {}, "labelIds": {}}}
    assert normalise_arguments({"event_id": "e1"}, schema)["eventId"] == "e1"
    assert normalise_arguments({"messageId": "m1"}, schema)["messageId"] == "m1"
    # snake_case → camelCase for anything declared
    assert normalise_arguments({"label_ids": ["INBOX"]}, schema)["labelIds"] == ["INBOX"]
    assert normalise_arguments({}, schema) == {}
    assert normalise_arguments({"x": None}, schema) == {}


def test_normalise_arguments_shapes_calendar_values():
    from otto.mcp.manager import normalise_arguments

    schema = {"properties": {"summary": {}, "startTime": {}, "endTime": {}, "attendees": {}, "attendeeEmails": {}}}
    got = normalise_arguments(
        {
            "summary": "Design sync",
            "start": {"dateTime": "2026-11-25T16:00:00-08:00", "timeZone": "America/Los_Angeles"},
            "end": {"dateTime": "2026-11-25T16:30:00-08:00"},
            "attendees": [{"email": "a@b.co"}, {"email": "c@d.co"}],
        },
        schema,
    )
    # REST-style nested objects become the plain strings the MCP tool wants
    assert got["startTime"] == "2026-11-25T16:00:00-08:00"
    assert got["endTime"] == "2026-11-25T16:30:00-08:00"
    assert got["attendeeEmails"] == ["a@b.co", "c@d.co"]
    assert "attendees" not in got


def test_normalise_arguments_keeps_all_day_dates():
    from otto.mcp.manager import normalise_arguments

    schema = {"properties": {"startTime": {}, "endTime": {}}}
    got = normalise_arguments({"start": {"date": "2026-11-30"}, "end": {"date": "2026-12-01"}}, schema)
    assert got == {"startTime": "2026-11-30", "endTime": "2026-12-01"}


async def test_content_level_error_is_not_reported_as_empty(tmp_path):
    """Google answers 200 with isError content for the preview gate — say so."""
    server = FakeMCPServer(
        tools=calendar_tools(),
        require_token="tok",
        results={
            "list_events": {
                "content": [{"type": "text", "text": "Access to this tool requires that your Google Cloud project (1) is enrolled in the Google Workspace Developer Preview Program."}],
                "isError": True,
            }
        },
    )
    manager = manager_for(tmp_path, server, name="calendar", servers=("calendar",))
    await manager.discover()
    tool = manager.get("calendar_list_events")
    result = await tool.invoke(ToolContext(settings=manager.settings), {"startTime": "2026-11-30T00:00:00Z"})
    assert result.ok is False
    assert "generally available" in (result.error or "")
    # the model is told the account is connected, so it cannot claim otherwise
    assert "IS connected" in result.summary
    assert "generally available" in result.summary
    assert "no calendar/mail access" in result.summary
    assert "calendar is clear" not in result.summary.lower()
    assert "No events" not in result.summary
    assert manager.status()["lastError"]["server"] == "calendar"
    await manager.aclose()


async def test_content_level_error_falls_back_to_rest(tmp_path):
    server = FakeMCPServer(
        tools=calendar_tools(),
        require_token="tok",
        results={
            "list_events": {
                "content": [{"type": "text", "text": "Access to this tool requires enrollment in the Google Workspace Developer Preview Program."}],
                "isError": True,
            }
        },
    )
    from otto.tools.base import Tool, ToolResult

    async def rest_handler(ctx, args):
        return ToolResult(ok=True, summary="3 events via REST", display={"kind": "event_list", "items": [{"id": "e1"}]})

    twin = Tool(name="calendar_list_events", description="d", parameters={"type": "object", "properties": {}}, handler=rest_handler)
    settings = settings_for(tmp_path, servers=("calendar",))
    manager = MCPManager(
        settings,
        token_provider=lambda: _value("tok"),
        transports={"calendar": httpx.ASGITransport(app=server.app)},
        fallback_lookup=lambda name: twin if name == "calendar_list_events" else None,
    )
    await manager.discover()
    result = await manager.get("calendar_list_events").invoke(ToolContext(settings=manager.settings), {"time_min": "2026-11-30"})
    assert result.ok is True
    assert result.summary == "3 events via REST"
    assert result.display["via"] == "rest"
    assert manager.status()["lastError"] is None, "a call the REST twin answered is not a refusal"
    await manager.aclose()


async def test_manager_falls_back_to_rest_when_mcp_refuses(tmp_path):
    """Google gates some tools behind the Developer Preview; the REST twin answers."""
    server = FakeMCPServer(
        tools=calendar_tools(),
        require_token="tok",
        forbid_calls=True,
        forbidden_detail="requires enrollment in the Google Workspace Developer Preview Program",
    )
    calls: list[dict] = []

    async def rest_handler(ctx, args):
        from otto.tools.base import ToolResult

        calls.append(args)
        return ToolResult(ok=True, summary="2 events (via REST)", display={"kind": "event_list", "items": [{"id": "e1"}]})

    from otto.tools.base import Tool

    rest_twin = Tool(
        name="calendar_list_events",
        description="rest calendar",
        parameters={"type": "object", "properties": {"time_min": {"type": "string"}}},
        handler=rest_handler,
    )
    settings = settings_for(tmp_path, servers=("calendar",))
    manager = MCPManager(
        settings,
        token_provider=lambda: _value("tok"),
        transports={"calendar": httpx.ASGITransport(app=server.app)},
        fallback_lookup=lambda name: rest_twin if name == "calendar_list_events" else None,
    )
    await manager.discover()
    tool = manager.get("calendar_list_events")
    result = await tool.invoke(ToolContext(settings=manager.settings), {"time_min": "2026-11-30T00:00:00Z"})
    assert result.ok
    assert result.summary == "2 events (via REST)"
    assert result.display["via"] == "rest"
    # the REST twin got the model's original arguments, untouched
    assert calls == [{"time_min": "2026-11-30T00:00:00Z"}]
    # the user is told once, not every call
    notices = manager.drain_notices()
    assert any("Developer Preview" in notice for notice in notices)
    await manager.aclose()


async def test_no_fallback_when_disabled(tmp_path):
    server = FakeMCPServer(tools=calendar_tools(), require_token="tok", forbid_calls=True)
    from otto.tools.base import Tool

    rest_twin = Tool(
        name="calendar_list_events",
        description="rest calendar",
        parameters={"type": "object", "properties": {}},
        handler=lambda ctx, args: None,
    )
    settings = settings_for(tmp_path, servers=("calendar",), fallback=False)
    manager = MCPManager(
        settings,
        token_provider=lambda: _value("tok"),
        transports={"calendar": httpx.ASGITransport(app=server.app)},
        fallback_lookup=lambda name: rest_twin,
    )
    await manager.discover()
    result = await manager.get("calendar_list_events").invoke(ToolContext(settings=manager.settings), {})
    assert result.ok is False
    assert "403" in (result.error or "") or "forbidden" in (result.error or "").lower()
    await manager.aclose()


def test_http_errors_explain_the_preview_gate():
    from otto.mcp.client import _explain_http

    sentence = _explain_http("Access to this tool requires that your Google Cloud project (1) is enrolled in the Google Workspace Developer Preview Program.")
    assert "Developer Preview" in sentence


def test_no_rest_probe_in_strict_mode(tmp_path):
    """Strict MCP: never suggest the classic APIs."""
    server = FakeMCPServer(tools=gmail_tools(), require_token="tok", forbid_calls=True,
                           forbidden_detail="requires enrollment in the Google Workspace Developer Preview Program")
    settings = settings_for(tmp_path, servers=("gmail",), fallback=False)
    manager = MCPManager(
        settings,
        token_provider=lambda: _value("tok"),
        transports={"gmail": httpx.ASGITransport(app=server.app)},
    )
    assert manager.authoritative is True


async def test_self_test_in_strict_mode_reports_only_mcp(tmp_path):
    server = FakeMCPServer(
        tools=gmail_tools() + [{
            "name": "list_labels", "title": "List labels", "description": "d",
            "inputSchema": {"type": "object", "properties": {}}, "annotations": {"readOnlyHint": True},
        }],
        require_token="tok",
        results={"list_labels": {"content": [{"type": "text", "text": "Developer Preview enrollment required"}], "isError": True}},
    )
    settings = settings_for(tmp_path, servers=("gmail",), fallback=False)
    manager = MCPManager(
        settings,
        token_provider=lambda: _value("tok"),
        transports={"gmail": httpx.ASGITransport(app=server.app)},
    )
    result = await manager.self_test(None)
    entry = result["servers"]["gmail"]
    assert entry["ok"] is False
    assert "generally available" in entry["detail"]
    assert "REST" not in entry["detail"]
    assert "calendar-json" not in entry["detail"]
    await manager.aclose()


# ------------------------------------------------------------------ harness
async def test_session_prefers_mcp_tools_over_rest(tmp_path):
    """The same name exists in both worlds; MCP must win and REST must disappear."""
    from .test_agents import RouterLLM, make_session

    server = FakeMCPServer(
        tools=gmail_tools(),
        require_token="tok",
        results={"search_threads": {"threads": [{"id": "t9", "subject": "MCP wins", "from": "a@b.co"}]}},
    )
    session, _ = make_session(tmp_path, llm=RouterLLM({}))
    session.mcp = manager_for(tmp_path, server)
    # force the REST twins to look available, so the name collision is real
    for name in ("gmail_search", "gmail_read", "gmail_draft"):
        tool = session.registry.get(name)
        if tool is not None:
            tool.available_for = lambda ctx: True

    schemas = await session._turn_tools()
    names = [schema["function"]["name"] for schema in schemas]
    assert "gmail_search" in names
    # MCP's gmail_search replaced the REST one (no duplicate name)
    assert names.count("gmail_search") == 1
    # in authoritative mode the legacy REST calendar tools are not offered
    if session.mcp.authoritative:
        assert "calendar_create_event" not in names
    await session.mcp.aclose()


async def _value(token: str) -> str:
    return token


def _unused(_: str) -> None:  # pragma: no cover
    return None


# keep flake8 happy about the asyncio import used by pytest-asyncio
_ = (asyncio, json)


async def _raw_status_app(status_by_method):
    """A raw ASGI MCP endpoint that can answer selected methods with an HTTP error."""
    import json as _json

    async def app(scope, receive, send):
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        payload = _json.loads(body or b"{}")
        method = payload.get("method", "")
        if method == "initialize":
            out = {"jsonrpc": "2.0", "id": payload.get("id"), "result": {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "serverInfo": {"name": "fake", "version": "0"}}}
            status = 200
        elif method in status_by_method:
            out, status = status_by_method[method]
        else:  # notifications/initialized, pings — accept and ignore
            out, status = {"jsonrpc": "2.0", "id": payload.get("id"), "result": {}}, 200
        data = _json.dumps(out).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": data})

    return app


async def test_client_keeps_jsonrpc_result_from_http_400():
    """Google rejects a bad tool argument with HTTP 400 *and* a JSON-RPC body — keep the message."""
    from otto.mcp.client import MCPClient

    async def token():
        return "tok"

    app = await _raw_status_app({
        "tools/call": ({"jsonrpc": "2.0", "id": 3, "result": {
            "content": [{"type": "text",
                        "text": 'Invalid JSON payload received. Unknown name "maxResults": Cannot find field.'}],
            "isError": True}}, 400),
    })
    client = MCPClient("http://testserver/mcp", name="gmail", token_provider=token,
                       transport=httpx.ASGITransport(app=app))
    await client.initialize()
    result = await client.call_tool("search_threads", {"query": "is:unread"})
    assert result.is_error is True
    assert 'Unknown name "maxResults"' in result.text
    assert "did not recognise" not in result.text
    await client.aclose()


async def test_client_raises_server_error_from_http_400():
    """A JSON-RPC error body on a 400 beats the generic status-code wording."""
    import pytest
    from otto.mcp.client import MCPClient, MCPError

    async def token():
        return "tok"

    app = await _raw_status_app({
        "tools/call": ({"jsonrpc": "2.0", "id": 3, "error": {
            "code": -32000, "message": "form elicitation required for this call"}}, 400),
    })
    client = MCPClient("http://testserver/mcp", name="calendar", token_provider=token,
                       transport=httpx.ASGITransport(app=app))
    await client.initialize()
    with pytest.raises(MCPError) as excinfo:
        await client.call_tool("update_event", {"eventId": "x"})
    assert "form elicitation required" in str(excinfo.value)
    await client.aclose()


async def test_mcp_tools_are_usable_without_a_context(tmp_path):
    """A voice turn asks `is_available()` with NO ToolContext — MCP tools must still pass.

    Regression: availability was wired as `available_for` (ctx-taking), which
    short-circuits to False without a context, so every spoken turn silently
    dropped all Gmail/Calendar tools and the model claimed it had no access.
    """
    server = FakeMCPServer(tools=gmail_tools(), require_token="tok")
    manager = manager_for(tmp_path, server, name="gmail", servers=("gmail",))
    tools = await manager.discover()
    assert tools, "discovery found nothing"
    assert all(tool.is_available() for tool in tools), "MCP tools unusable without a ctx"
    assert all(tool.is_available(ToolContext(settings=manager.settings)) for tool in tools)
    await manager.aclose()


async def test_rest_fallback_translates_mcp_arg_names(tmp_path):
    """The model speaks MCP (`startTime`); the REST twin declares `time_min`."""
    server = FakeMCPServer(
        tools=calendar_tools(),
        require_token="tok",
        results={"list_events": {
            "content": [{"type": "text", "text": "Access requires enrollment in the Google Workspace Developer Preview Program."}],
            "isError": True,
        }},
    )
    from otto.tools.base import Tool

    seen: dict[str, Any] = {}

    async def rest_handler(ctx, args):
        from otto.tools.base import ToolResult
        seen.update(args)
        return ToolResult(ok=True, summary="ok", display={"kind": "event_list", "items": []})

    twin = Tool(
        name="calendar_list_events",
        description="d",
        parameters={"type": "object", "properties": {
            "time_min": {"type": "string"}, "time_max": {"type": "string"}, "max_results": {"type": "integer"}}},
        handler=rest_handler,
    )
    settings = settings_for(tmp_path, servers=("calendar",))
    manager = MCPManager(
        settings,
        token_provider=lambda: _value("tok"),
        transports={"calendar": httpx.ASGITransport(app=server.app)},
        fallback_lookup=lambda name: twin if name == "calendar_list_events" else None,
    )
    await manager.discover()
    result = await manager.get("calendar_list_events").invoke(
        ToolContext(settings=manager.settings),
        {"startTime": "2026-11-30T00:00:00-08:00", "endTime": "2026-12-01T00:00:00-08:00",
         "pageSize": 10, "orderBy": "startTime", "timeZone": "America/Los_Angeles"},
    )
    assert result.ok is True
    assert seen["time_min"].startswith("2026-11-30T00:00:00-08:00")
    assert seen["time_max"].startswith("2026-12-01")
    assert seen["max_results"] == 10
    assert "startTime" not in seen and "pageSize" not in seen and "orderBy" not in seen
    await manager.aclose()


async def test_create_event_note_reaches_the_model(tmp_path):
    """The model is told attendees are optional so it never demands an email."""
    server = FakeMCPServer(tools=calendar_tools(), require_token="tok")
    manager = manager_for(tmp_path, server, name="calendar", servers=("calendar",))
    await manager.discover()
    for name in ("calendar_create_event", "calendar_update_event"):
        tool = manager.get(name)
        assert tool is not None, name
        assert "Attendees are optional" in tool.description
        assert "never ask" in tool.description.lower()
        assert "overrideReminders" in tool.description, "the model must know it can change reminders"
    await manager.aclose()


async def test_rest_fallback_maps_create_args_too(tmp_path):
    """`startTime` is `time_min` on the list twin but `start` on the create twin."""
    server = FakeMCPServer(
        tools=calendar_tools(),
        require_token="tok",
        results={"create_event": {
            "content": [{"type": "text", "text": "Access requires enrollment in the Google Workspace Developer Preview Program."}],
            "isError": True,
        }},
    )
    from otto.tools.base import Tool

    seen: dict = {}

    async def rest_handler(ctx, args):
        from otto.tools.base import ToolResult
        seen.update(args)
        return ToolResult(ok=True, summary="created", display={"kind": "event_created", "item": {"id": "e1"}})

    twin = Tool(
        name="calendar_create_event",
        description="d",
        parameters={"type": "object", "properties": {
            "summary": {"type": "string"}, "start": {"type": "string"},
            "end": {"type": "string"}, "attendees": {"type": "array", "items": {"type": "string"}},
            "reminders": {"type": "array"}, "use_default_reminders": {"type": "boolean"}}},
        handler=rest_handler,
    )
    settings = settings_for(tmp_path, servers=("calendar",))
    manager = MCPManager(
        settings,
        token_provider=lambda: _value("tok"),
        transports={"calendar": httpx.ASKITransport(app=server.app)} if False else {"calendar": httpx.ASGITransport(app=server.app)},
        fallback_lookup=lambda name: twin if name == "calendar_create_event" else None,
    )
    await manager.discover()
    result = await manager.get("calendar_create_event").invoke(
        ToolContext(settings=manager.settings),
        {"summary": "Meeting with boss", "startTime": "2026-11-26T15:00:00", "endTime": "2026-11-26T15:30:00",
         "overrideReminders": [{"method": "popup", "minutes": 10}]},
    )
    assert result.ok is True
    assert seen == {"summary": "Meeting with boss", "start": "2026-11-26T15:00:00", "end": "2026-11-26T15:30:00",
                    "reminders": [{"method": "popup", "minutes": 10}]}
    await manager.aclose()


def test_update_prompt_speaks_changes_not_ids():
    """The approval question describes the change in human words."""
    from otto.mcp.policy import _event_update_prompt

    # a reminders-only change: say what changes, never offer a rename
    prompt = _event_update_prompt({
        "eventId": "ku5mdqp4bmnjc66ppqvl8juv6o",
        "overrideReminders": [{"method": "popup", "minutes": 10}],
    })
    assert "popup reminder 10 minutes before" in prompt
    assert "rename" not in prompt
    assert "ku5mdqp4bmnjc66ppqvl8juv6o" in prompt  # the session swaps it for the title

    # a pure rename
    prompt = _event_update_prompt({"eventId": "e1", "summary": "Root canal"})
    assert prompt.startswith("I'll rename") and "'Root canal'" in prompt

    # a move — the timestamp is spoken like a person, not an ISO string
    prompt = _event_update_prompt({"eventId": "e1", "startTime": "2026-11-26T15:00:00-08:00"})
    assert "move it to Thursday, November 26 at 3 PM" in prompt

    # the model passing the current title as an identifier must not read as a rename
    prompt = _event_update_prompt({
        "eventId": "e1", "summary": "Otto reminder test",
        "overrideReminders": [{"method": "popup", "minutes": 10}],
    })
    assert "rename" not in prompt
