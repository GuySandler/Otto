"""Small local tools: clock, and a note pad the assistant can use as memory."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..settings import get_current
from .base import Tool, ToolContext, ToolResult


def _notes_file() -> Path:
    return get_current().data_dir / "notes.json"


def _now_payload() -> dict[str, Any]:
    now = datetime.now().astimezone()
    return {
        "iso": now.isoformat(timespec="seconds"),
        "local": now.strftime("%A, %B %d %Y at %I:%M %p").replace(" 0", " "),
        "weekday": now.strftime("%A"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "timezone": now.tzname() or "",
        "utc_offset": now.strftime("%z"),
    }


async def _get_time(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    payload = _now_payload()
    return ToolResult(
        ok=True,
        summary=(
            f"It is {payload['local']} ({payload['timezone']}, UTC{payload['utc_offset'][:3]}:{payload['utc_offset'][3:]}). "
            f"ISO: {payload['iso']}"
        ),
        display={"kind": "text", "text": payload["local"]},
        data=payload,
    )


async def _remember(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    text = str(args.get("text") or "").strip()
    if not text:
        return ToolResult(ok=False, error="nothing to remember")
    path = _notes_file()
    notes: list[dict[str, Any]] = []
    if path.exists():
        try:
            notes = json.loads(path.read_text())
        except ValueError:
            notes = []
    notes.append({"at": _now_payload()["iso"], "text": text})
    path.write_text(json.dumps(notes, indent=2, ensure_ascii=False))
    return ToolResult(
        ok=True,
        summary=f"Noted: {text}",
        display={"kind": "text", "text": f"Noted: {text}"},
    )


async def _recall(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    path = _notes_file()
    query = str(args.get("query") or "").lower().strip()
    notes: list[dict[str, Any]] = []
    if path.exists():
        try:
            notes = json.loads(path.read_text())
        except ValueError:
            notes = []
    if query:
        notes = [n for n in notes if query in str(n.get("text", "")).lower()]
    if not notes:
        return ToolResult(ok=True, summary="No matching notes.", display={"kind": "text", "text": "No notes."})
    recent = notes[-10:]
    summary = "; ".join(f"{n['text']} (noted {n['at'][:16]})" for n in recent)
    return ToolResult(
        ok=True,
        summary=summary,
        display={"kind": "list", "items": [n["text"] for n in reversed(recent)]},
    )


TOOLS: list[Tool] = [
    Tool(
        name="get_current_time",
        description=(
            "Get the current local date and time, weekday and timezone. Always call this "
            "before creating or interpreting calendar events expressed relatively "
            "('tomorrow', 'next Tuesday', 'in an hour')."
        ),
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_get_time,
        category="system",
        timeout_s=5.0,
    ),
    Tool(
        name="remember",
        description="Store a short note or fact for later recall in this session's notes file.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The note to store."}},
            "required": ["text"],
        },
        handler=_remember,
        category="system",
        timeout_s=5.0,
    ),
    Tool(
        name="recall",
        description="Look up previously stored notes, optionally filtered by a keyword.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Optional keyword filter."}},
        },
        handler=_recall,
        category="system",
        timeout_s=5.0,
    ),
]
