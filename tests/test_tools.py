"""Tool registry, catalog and confirmation plumbing."""
from __future__ import annotations

import json

import pytest

from otto.tools.base import Tool, ToolContext, ToolResult
from otto.tools.registry import ToolRegistry


def _tool(name: str, **kwargs) -> Tool:
    async def handler(ctx, args):
        return ToolResult(ok=True, summary=f"{name} ok")

    return Tool(
        name=name,
        description=f"{name} does things",
        parameters={"type": "object", "properties": {}},
        handler=handler,
        **kwargs,
    )


def test_discovery_finds_bundled_modules():
    registry = ToolRegistry.discover()
    names = set(registry.names())
    assert {"gmail_search", "gmail_send", "calendar_create_event", "slack_send_message"} <= names
    assert "get_current_time" in names


def test_schemas_exclude_unconfigured_tools():
    registry = ToolRegistry()
    registry.add(_tool("always_on"))
    registry.add(_tool("needs_config", available=lambda: False, unavailable_hint="no creds"))
    assert [s["function"]["name"] for s in registry.schemas()] == ["always_on"]
    catalog = {c["name"]: c for c in registry.catalog()}
    assert catalog["needs_config"]["configured"] is False
    assert catalog["needs_config"]["unavailable_hint"] == "no creds"


def test_confirmation_flag_predicate_and_prompt():
    tool = _tool(
        "maybe_danger",
        requires_confirmation=lambda args: bool(args.get("force")),
        confirmation_prompt=lambda args: f"force={args.get('force')}?",
    )
    assert tool.needs_confirmation({}) is False
    assert tool.needs_confirmation({"force": True}) is True
    assert tool.confirmation_text({"force": True}) == "force=True?"


def test_confirmation_defaults_to_a_sentence():
    tool = _tool("send_money", requires_confirmation=True)
    assert "send money" in tool.confirmation_text({"amount": 5})


async def test_tool_invoke_captures_errors_and_timeouts():
    async def boom(ctx, args):
        raise ValueError("nope")

    broken = Tool(
        name="broken",
        description="d",
        parameters={"type": "object", "properties": {}},
        handler=boom,
    )
    result = await broken.invoke(ToolContext(settings=None), {})
    assert result.ok is False
    assert "nope" in (result.error or "")

    async def slow(ctx, args):
        import asyncio

        await asyncio.sleep(1)
        return ToolResult(ok=True)

    slow_tool = Tool(
        name="slow",
        description="d",
        parameters={"type": "object", "properties": {}},
        handler=slow,
        timeout_s=0.05,
    )
    timed_out = await slow_tool.invoke(ToolContext(settings=None), {})
    assert timed_out.ok is False
    assert "timed out" in (timed_out.error or "")


async def test_tool_result_payload_keeps_small_data_only():
    big = ToolResult(ok=True, summary="fine", data={"blob": "x" * 20000})
    payload = json.loads(big.to_model_payload(limit=200))
    assert payload["summary"] == "fine"
    assert "data" not in payload

    small = ToolResult(ok=True, summary="fine", data={"n": 1})
    assert json.loads(small.to_model_payload())["data"] == {"n": 1}


def test_duplicate_tool_names_are_ignored():
    registry = ToolRegistry()
    registry.add(_tool("dup"))
    registry.add(_tool("dup"))
    assert registry.names().count("dup") == 1


def test_plugin_files_are_loaded(tmp_path):
    plugin = tmp_path / "my_tool.py"
    plugin.write_text(
        "from otto.tools.base import Tool, ToolResult\n"
        "async def handler(ctx, args):\n"
        "    return ToolResult(ok=True, summary='plugin')\n"
        "TOOLS = [Tool(name='plugin_tool', description='d', "
        "parameters={'type': 'object', 'properties': {}}, handler=handler)]\n"
    )
    registry = ToolRegistry.discover(plugin_paths=[str(plugin)])
    assert "plugin_tool" in registry.names()


def test_missing_plugin_is_not_fatal():
    registry = ToolRegistry.discover(plugin_paths=["/nonexistent/tool.py"])
    assert "get_current_time" in registry.names()


def test_gmail_send_requires_the_send_scope():
    """`gmail_send` is hidden until the consent actually carries the gmail.send scope."""
    from otto.tools.gmail import TOOLS

    class Google:
        connected = True
        scopes = ["https://www.googleapis.com/auth/gmail.modify"]

    class Integrations:
        google = Google()

    tool = next(t for t in TOOLS if t.name == "gmail_send")
    ctx = ToolContext(settings=None, integrations=Integrations())
    assert tool.is_available(ctx) is False, "visible without the send scope"

    Google.scopes = Google.scopes + ["https://www.googleapis.com/auth/gmail.send"]
    assert tool.is_available(ctx) is True

    Google.scopes = []
    Google.connected = False
    assert tool.is_available(ctx) is False


def test_people_tools_profile_caches_the_name():
    """people_profile remembers the name so the next session can sign drafts."""
    from otto.tools.people import TOOLS

    class Google:
        connected = True

        async def people_profile(self):
            return {"name": "Guy Sandler", "email": "guy@sandler.co.il"}

        async def people_search(self, query):
            return [{"name": "David Byrne", "email": "david@boss.com", "emails": ["david@boss.com"]}]

    class Users:
        saved: dict = {}

        def set_secret(self, user_id, key, value):
            self.saved[(user_id, key)] = value

        def get_secret(self, user_id, key):
            return self.saved.get((user_id, key))

    class Integrations:
        google = Google()
        users = Users()
        user_id = "u1"

    ctx = ToolContext(settings=None, integrations=Integrations())
    profile = next(t for t in TOOLS if t.name == "people_profile")
    search = next(t for t in TOOLS if t.name == "people_search")

    assert profile.is_available(ctx) and search.is_available(ctx)

    import asyncio

    result = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(profile.invoke(ctx, {}))
    assert result.ok
    assert "Guy Sandler" in result.summary
    assert Integrations.users.saved[("u1", "profile_name")] == "Guy Sandler"

    result = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(search.invoke(ctx, {"query": "david"}))
    assert result.ok and "david@boss.com" in result.summary
