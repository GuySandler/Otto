"""Slack tools — read, search, send (send asks first)."""
from __future__ import annotations

from typing import Any

from ..settings import get_current
from .base import Tool, ToolContext, ToolResult


def _slack(ctx: ToolContext):
    return ctx.integrations.slack


def _ready(ctx: ToolContext) -> bool:
    """Per-session check: does this signed-in account have a Slack token?"""
    try:
        return bool(ctx.integrations.slack.configured)
    except Exception:  # noqa: BLE001
        return False


def _short(text: str, limit: int = 200) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _card_message(message: dict[str, Any], channel: str) -> dict[str, Any]:
    return {
        "channel": channel,
        "user": message.get("user_name") or message.get("user", ""),
        "text": _short(message.get("text", ""), 600),
        "ts": message.get("ts", ""),
        "permalink": message.get("permalink", ""),
        "replies": message.get("replies") or 0,
    }


async def _channels(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    channels = await _slack(ctx).channels(limit=min(int(args.get("limit") or 100), 400))
    if not channels:
        return ToolResult(ok=True, summary="No channels visible to this bot.", display={"kind": "list", "items": []})
    names = [c["name"] for c in channels]
    return ToolResult(
        ok=True,
        summary="Channels: " + ", ".join(names[:60]),
        display={"kind": "slack_channels", "items": channels},
    )


async def _read(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    channel = str(args.get("channel") or "").strip()
    if not channel:
        channel = get_current().slack_default_channel
    if not channel:
        return ToolResult(ok=False, error="channel is required (name like 'general' or a channel id)")
    limit = min(int(args.get("limit") or 15), 50)
    resolved = await _slack(ctx).resolve_channel(channel)
    messages = await _slack(ctx).history(resolved, limit=limit)
    if not messages:
        return ToolResult(ok=True, summary=f"No messages in {channel}.", display={"kind": "slack_messages", "channel": channel, "items": []})
    lines = [f"- {m.get('user_name') or m.get('user')}: {_short(m.get('text', ''), 220)}" for m in messages]
    return ToolResult(
        ok=True,
        summary=f"Last {len(messages)} message(s) in {channel}:\n" + "\n".join(lines),
        display={
            "kind": "slack_messages",
            "channel": channel,
            "items": [_card_message(m, channel) for m in messages],
        },
    )


async def _search(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    query = str(args.get("query") or "").strip()
    if not query:
        return ToolResult(ok=False, error="query is required")
    results = await _slack(ctx).search(query, count=min(int(args.get("count") or 10), 25))
    if not results:
        return ToolResult(
            ok=True,
            summary=f"No Slack messages matched '{query}'.",
            display={"kind": "slack_messages", "channel": "", "items": []},
        )
    lines = [
        f"- #{m.get('channel_name', m.get('channel', ''))} | {m.get('user_name') or m.get('user')} | {_short(m.get('text', ''), 220)}"
        for m in results
    ]
    return ToolResult(
        ok=True,
        summary=f"{len(results)} Slack message(s) for '{query}':\n" + "\n".join(lines),
        display={
            "kind": "slack_messages",
            "channel": "",
            "items": [_card_message(m, m.get("channel_name", "")) for m in results],
        },
    )


def _send_prompt(args: dict[str, Any]) -> str:
    channel = str(args.get("channel") or "the channel")
    text = _short(str(args.get("text") or ""), 140)
    return f"I'll post to {channel}: {text}. Send it?"


def _send_preview(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "slack_send",
        "channel": str(args.get("channel") or ""),
        "text": str(args.get("text") or ""),
    }


async def _send(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    channel = str(args.get("channel") or "").strip() or get_current().slack_default_channel
    text = str(args.get("text") or "").strip()
    if not channel:
        return ToolResult(ok=False, error="channel is required")
    if not text:
        return ToolResult(ok=False, error="text is required")
    resolved = await _slack(ctx).resolve_channel(channel)
    result = await _slack(ctx).post_message(resolved, text, thread_ts=args.get("thread_ts"))
    return ToolResult(
        ok=True,
        summary=f"Posted to {channel}.",
        display={
            "kind": "slack_sent",
            "channel": channel,
            "text": text,
            "ts": result.get("ts", ""),
        },
    )


TOOLS: list[Tool] = [
    Tool(
        name="slack_list_channels",
        description="List Slack channels this bot can see (use it to turn a channel name into an id).",
        parameters={
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Max channels (default 100)."}},
        },
        handler=_channels,
        category="slack",
        available_for=_ready,
        unavailable_hint="Set SLACK_BOT_TOKEN in .env to use Slack.",
    ),
    Tool(
        name="slack_read_channel",
        description="Read the most recent messages in a Slack channel, newest last.",
        parameters={
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Channel name ('general', '#general') or id."},
                "limit": {"type": "integer", "description": "How many messages (default 15, max 50)."},
            },
            "required": ["channel"],
        },
        handler=_read,
        category="slack",
        available_for=_ready,
        unavailable_hint="Set SLACK_BOT_TOKEN in .env to use Slack.",
    ),
    Tool(
        name="slack_search",
        description=(
            "Search Slack messages across channels. Supports Slack search modifiers like "
            "'from:@sarah', 'in:#design', 'has:link'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "count": {"type": "integer", "description": "Max results (default 10)."},
            },
            "required": ["query"],
        },
        handler=_search,
        category="slack",
        available_for=_ready,
        unavailable_hint="Set SLACK_BOT_TOKEN (with search:read) in .env to use Slack search.",
    ),
    Tool(
        name="slack_send_message",
        description=(
            "Post a message to a Slack channel or reply in a thread. The user is asked to "
            "confirm before it is posted — call it with the final text."
        ),
        parameters={
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Channel name or id."},
                "text": {"type": "string", "description": "Message text (Slack markdown allowed)."},
                "thread_ts": {"type": "string", "description": "Optional thread timestamp to reply in a thread."},
            },
            "required": ["channel", "text"],
        },
        handler=_send,
        category="slack",
        requires_confirmation=True,
        confirmation_prompt=_send_prompt,
        preview=_send_preview,
        available_for=_ready,
        unavailable_hint="Set SLACK_BOT_TOKEN in .env to use Slack.",
    ),
]
