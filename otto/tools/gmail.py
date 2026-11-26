"""Gmail tools — read, search, draft, send (send asks first)."""
from __future__ import annotations

from typing import Any

from ..settings import get_current
from .base import Tool, ToolContext, ToolResult


def _google(ctx: ToolContext):
    return ctx.integrations.google


def _ready(ctx: ToolContext) -> bool:
    """Per-session check: did *this* signed-in account connect Google?"""
    try:
        return bool(ctx.integrations.google.connected)
    except Exception:  # noqa: BLE001
        return False


def _send_ready(ctx: ToolContext) -> bool:
    """Sending needs the `gmail.send` scope on top of a connected account.

    The scope was dropped from the default consent while the MCP-only phase
    required it not to exist; it comes back the moment the user re-adds it to
    `google.scopes` and reconnects.
    """
    if not _ready(ctx):
        return False
    try:
        scopes = list(ctx.integrations.google.scopes or [])
    except Exception:  # noqa: BLE001
        return False
    return any(str(scope).rstrip("/").endswith("auth/gmail.send") for scope in scopes)


def _fmt_addr(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value or "")


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _card(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": message.get("id", ""),
        "from": _fmt_addr(message.get("from_") or message.get("from")),
        "to": _fmt_addr(message.get("to")),
        "subject": message.get("subject", ""),
        "date": message.get("date", ""),
        "snippet": _short(message.get("snippet", ""), 180),
        "unread": bool(message.get("unread")),
    }


async def _search(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    query = str(args.get("query") or "").strip()
    max_results = min(int(args.get("max_results") or 8), 25)
    messages = await _google(ctx).gmail_search(query, max_results=max_results)
    if not messages:
        return ToolResult(
            ok=True,
            summary=f"No emails matched '{query}'.",
            display={"kind": "email_list", "query": query, "items": []},
        )
    lines = []
    for msg in messages:
        unread = " (unread)" if msg.get("unread") else ""
        lines.append(
            f"- {msg.get('id')} | from {_fmt_addr(msg.get('from_'))} | {msg.get('subject')} | "
            f"{msg.get('date')}{unread} | {_short(msg.get('snippet', ''), 120)}"
        )
    summary = f"{len(messages)} email(s) for query '{query}':\n" + "\n".join(lines)
    return ToolResult(
        ok=True,
        summary=summary,
        display={"kind": "email_list", "query": query, "items": [_card(m) for m in messages]},
    )


async def _read(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    message_id = str(args.get("message_id") or "").strip()
    if not message_id:
        return ToolResult(ok=False, error="message_id is required")
    message = await _google(ctx).gmail_get(message_id)
    body = _short(message.get("body_text", ""), 1800)
    summary = (
        f"From: {_fmt_addr(message.get('from_'))}\n"
        f"To: {_fmt_addr(message.get('to'))}\n"
        f"Subject: {message.get('subject')}\n"
        f"Date: {message.get('date')}\n\n{body}"
    )
    if message.get("attachments"):
        names = ", ".join(a.get("filename", "?") for a in message["attachments"][:5])
        summary += f"\n\nAttachments: {names}"
    card = _card(message)
    card["body"] = _short(message.get("body_text", ""), 4000)
    card["attachments"] = message.get("attachments", [])
    return ToolResult(ok=True, summary=summary, display={"kind": "email", "item": card})


async def _draft(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    to = [str(t).strip() for t in (args.get("to") or []) if str(t).strip()]
    if isinstance(args.get("to"), str):
        to = [t.strip() for t in str(args["to"]).split(",") if t.strip()]
    subject = str(args.get("subject") or "").strip()
    body = str(args.get("body") or "").strip()
    if not to or not subject:
        return ToolResult(ok=False, error="both 'to' and 'subject' are required")
    result = await _google(ctx).gmail_draft(to=to, subject=subject, body=body, cc=args.get("cc") or ())
    display = {
        "kind": "email_draft",
        "draftId": result.get("id", ""),
        "to": to,
        "subject": subject,
        "body": body,
    }
    return ToolResult(
        ok=True,
        summary=f"Draft created (id {result.get('id')}) to {', '.join(to)} with subject '{subject}'.",
        display=display,
    )


def _send_prompt(args: dict[str, Any]) -> str:
    to = args.get("to")
    if isinstance(to, str):
        to = [t.strip() for t in to.split(",") if t.strip()]
    recipients = ", ".join(str(t) for t in (to or [])) or "someone"
    subject = str(args.get("subject") or "").strip() or "no subject"
    return f"I've got an email to {recipients} about {subject}. Want me to send it?"


def _send_preview(args: dict[str, Any]) -> dict[str, Any]:
    to = args.get("to")
    if isinstance(to, str):
        to = [t.strip() for t in to.split(",") if t.strip()]
    return {
        "kind": "email_send",
        "to": to or [],
        "subject": str(args.get("subject") or ""),
        "body": str(args.get("body") or ""),
    }


async def _send(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    to_raw = args.get("to")
    to = [t.strip() for t in (to_raw.split(",") if isinstance(to_raw, str) else (to_raw or [])) if str(t).strip()]
    subject = str(args.get("subject") or "").strip()
    body = str(args.get("body") or "").strip()
    if not to or not subject:
        return ToolResult(ok=False, error="both 'to' and 'subject' are required")
    result = await _google(ctx).gmail_send(to=to, subject=subject, body=body, cc=args.get("cc") or ())
    return ToolResult(
        ok=True,
        summary=f"Email sent to {', '.join(to)} (subject '{subject}').",
        display={
            "kind": "email_sent",
            "to": to,
            "subject": subject,
            "body": body,
            "messageId": result.get("id", ""),
        },
    )


TOOLS: list[Tool] = [
    Tool(
        name="gmail_search",
        description=(
            "Search the user's Gmail and return matching messages (id, sender, subject, "
            "date, snippet). Supports Gmail search syntax, e.g. 'from:sarah newer_than:7d', "
            "'is:unread', 'subject:invoice', 'has:attachment'. Use this to answer "
            "'do I have mail from X' or 'what's unread'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Gmail search query. Empty string returns the newest mail."},
                "max_results": {"type": "integer", "description": "How many messages to return (default 8, max 25)."},
            },
            "required": ["query"],
        },
        handler=_search,
        category="gmail",
        available_for=_ready,
        unavailable_hint="Connect a Google account (scripts/google_auth.py) to use Gmail.",
    ),
    Tool(
        name="gmail_read",
        description="Read one email in full by its message id (get the id from gmail_search).",
        parameters={
            "type": "object",
            "properties": {"message_id": {"type": "string", "description": "Gmail message id."}},
            "required": ["message_id"],
        },
        handler=_read,
        category="gmail",
        available_for=_ready,
        unavailable_hint="Connect a Google account to use Gmail.",
    ),
    Tool(
        name="gmail_draft",
        description=(
            "Create a draft email without sending it. Use this when the user says 'draft', "
            "'write' or 'compose'. Nothing leaves the account until it is sent."
        ),
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "array", "items": {"type": "string"}, "description": "Recipient email addresses."},
                "subject": {"type": "string"},
                "body": {"type": "string", "description": "Plain-text body, no markdown."},
                "cc": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["to", "subject", "body"],
        },
        handler=_draft,
        category="gmail",
        available_for=_ready,
        unavailable_hint="Connect a Google account to use Gmail.",
    ),
    Tool(
        name="gmail_send",
        description=(
            "Send an email. This is irreversible, so the user is asked to confirm before it "
            "is delivered — just call it with the final text and let the system ask."
        ),
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "array", "items": {"type": "string"}, "description": "Recipient email addresses."},
                "subject": {"type": "string"},
                "body": {"type": "string", "description": "Plain-text body, no markdown."},
                "cc": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["to", "subject", "body"],
        },
        handler=_send,
        category="gmail",
        requires_confirmation=True,
        confirmation_prompt=_send_prompt,
        preview=_send_preview,
        available_for=_send_ready,
        unavailable_hint="Sending needs the gmail.send scope — add it to google.scopes and reconnect.",
    ),
]
