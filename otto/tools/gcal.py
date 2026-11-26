"""Google Calendar tools — read freely, create and delete with confirmation."""
from __future__ import annotations

from typing import Any

from ..memory import KIND_EVENT
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


def _resolve_event(ctx: ToolContext, args: dict[str, Any]) -> tuple[str | None, str]:
    """Work out which event the user means — explicit id, or "that thing" from memory."""
    event_id = str(args.get("event_id") or "").strip()
    hint = str(args.get("hint") or args.get("summary") or "").strip()
    if event_id:
        return event_id, hint or event_id
    session = getattr(ctx, "session", None)
    memory = getattr(session, "memory", None)
    if memory is not None:
        entity = memory.resolve(KIND_EVENT, hint or None)
        if entity is not None:
            return entity.id, entity.label
    return None, hint


def _when(event: dict[str, Any]) -> str:
    start = str(event.get("start") or "")
    end = str(event.get("end") or "")
    if event.get("all_day"):
        return f"all day {start[:10]}"
    if "T" in start:
        return f"{start[:16].replace('T', ' ')} → {end[11:16]}" if "T" in end else start[:16].replace("T", " ")
    return start


def _card(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": event.get("id", ""),
        "title": event.get("summary") or "(no title)",
        "start": event.get("start"),
        "end": event.get("end"),
        "all_day": bool(event.get("all_day")),
        "location": event.get("location", ""),
        "attendees": event.get("attendees", []),
        "link": event.get("html_link", ""),
    }


async def _list(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    events = await _google(ctx).calendar_list(
        time_min=args.get("time_min"),
        time_max=args.get("time_max"),
        query=args.get("query"),
        max_results=min(int(args.get("max_results") or 15), 50),
    )
    if not events:
        return ToolResult(
            ok=True,
            summary="No events in that window.",
            display={"kind": "event_list", "items": []},
        )
    lines = []
    for event in events:
        extras = []
        if event.get("location"):
            extras.append(str(event["location"]))
        if event.get("attendees"):
            extras.append(f"{len(event['attendees'])} attendees")
        suffix = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"- {event.get('id')} | {event.get('summary') or '(no title)'} | {_when(event)}{suffix}")
    return ToolResult(
        ok=True,
        summary=f"{len(events)} event(s):\n" + "\n".join(lines),
        display={"kind": "event_list", "items": [_card(e) for e in events]},
    )


async def _search(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """The REST twin of the MCP search_events tool — text search across a wide window."""
    query = str(args.get("query") or args.get("q") or "").strip()
    if not query:
        return ToolResult(ok=False, error="a search query is required")
    from datetime import datetime, timedelta

    now = datetime.now().astimezone()
    start = str(args.get("time_min") or (now - timedelta(days=30)).isoformat(timespec="seconds"))
    end = str(args.get("time_max") or (now + timedelta(days=60)).isoformat(timespec="seconds"))
    events = await _google(ctx).calendar_list(
        time_min=start, time_max=end, query=query,
        max_results=min(int(args.get("max_results") or 20), 50),
    )
    if not events:
        return ToolResult(ok=True, summary=f"No events matched '{query}'.",
                          display={"kind": "event_list", "items": []})
    lines = [f"- {e.get('id')} | {e.get('summary') or '(no title)'} | {_when(e)}" for e in events]
    return ToolResult(
        ok=True,
        summary=f"{len(events)} event(s) matching '{query}':\n" + "\n".join(lines),
        display={"kind": "event_list", "items": [_card(e) for e in events]},
    )


def _spoken_when(value: str) -> str:
    """'2026-11-26T15:00:00-08:00' → 'Friday, November 26 at 3 PM'."""
    text = str(value or "").strip()
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text[:16].replace("T", " at ") if "T" in text else text
    hour = moment.strftime("%I").lstrip("0") or "12"
    minute = moment.strftime("%M")
    clock = f"{hour}:{minute}" if minute != "00" else hour
    return f"{moment.strftime('%A, %B')} {moment.day} at {clock} {moment.strftime('%p')}"


def _create_prompt(args: dict[str, Any]) -> str:
    summary = str(args.get("summary") or "an event")
    start = str(args.get("start") or "")
    when = _spoken_when(start) if start else "that time"
    return f"I'll put {summary} on the calendar for {when}. Should I add it?"


def _create_preview(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "event_create",
        "title": str(args.get("summary") or ""),
        "start": args.get("start"),
        "end": args.get("end"),
        "location": str(args.get("location") or ""),
        "description": str(args.get("description") or ""),
        "attendees": args.get("attendees") or [],
    }


def _bool_or_none(value: Any) -> bool | None:
    # 'true'-ish strings count as True; missing stays missing
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


async def _create(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    summary = str(args.get("summary") or "").strip()
    start = str(args.get("start") or "").strip()
    end = str(args.get("end") or "").strip()
    if not summary or not start or not end:
        return ToolResult(ok=False, error="summary, start and end are required (RFC3339 datetimes)")
    event = await _google(ctx).calendar_create(
        summary=summary,
        start=start,
        end=end,
        description=str(args.get("description") or ""),
        location=str(args.get("location") or ""),
        attendees=args.get("attendees") or (),
        reminders=args.get("reminders") or None,
        use_default_reminders=_bool_or_none(args.get("use_default_reminders")),
    )
    return ToolResult(
        ok=True,
        summary=f"Created '{summary}' at {_when(event)} (id {event.get('id')}).",
        display={"kind": "event_created", "item": _card(event)},
    )


def _delete_prompt(args: dict[str, Any]) -> str:
    title = str(args.get("summary") or args.get("event_id") or "that event")
    return f"Do you want me to delete {title} from your calendar?"


def _delete_preview(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "event_delete",
        "id": str(args.get("event_id") or ""),
        "title": str(args.get("summary") or ""),
    }


async def _delete(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    event_id, label = _resolve_event(ctx, args)
    if not event_id:
        return ToolResult(
            ok=False,
            error="I don't know which event you mean. Ask which one, or list the calendar first.",
        )
    await _google(ctx).calendar_delete(event_id)
    title = str(args.get("summary") or label or event_id)
    return ToolResult(
        ok=True,
        summary=f"Deleted '{title}'.",
        display={"kind": "event_deleted", "id": event_id, "title": title},
    )


def _update_prompt(args: dict[str, Any]) -> str:
    target = str(args.get("event_id") or args.get("hint") or "that event")
    bits = []
    if args.get("start"):
        bits.append(f"move it to {_spoken_when(str(args['start']))}")
    if args.get("end"):
        bits.append(f"end at {_spoken_when(str(args['end']))}")
    if args.get("location"):
        bits.append(f"set the location to {args['location']}")
    reminders = args.get("reminders")
    if isinstance(reminders, list) and reminders:
        parts = []
        for item in reminders:
            if isinstance(item, dict):
                method = str(item.get("method") or "popup")
                article = "an" if method == "email" else "a"
                parts.append(f"{article} {method} reminder {item.get('minutes')} minutes before")
        if parts:
            bits.append("set " + " and ".join(parts))
    if args.get("use_default_reminders"):
        bits.append("restore the calendar's default reminders")
    if bits:
        return f"I'll update {target}: {', '.join(bits[:3])} — okay?"
    if args.get("summary"):
        return f"I'll rename {target} to '{args['summary']}' — okay?"
    return f"I'll update {target} — okay?"


def _update_preview(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "event_update",
        "id": str(args.get("event_id") or ""),
        "title": str(args.get("summary") or args.get("hint") or ""),
        "start": args.get("start"),
        "end": args.get("end"),
        "location": str(args.get("location") or ""),
    }


async def _update(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    event_id, label = _resolve_event(ctx, args)
    if not event_id:
        return ToolResult(
            ok=False,
            error=(
                "I don't know which event you mean. Ask the user which one, or list the calendar "
                "first so an id is available."
            ),
            summary="No matching event.",
        )
    if not any(
        args.get(key)
        for key in ("summary", "start", "end", "location", "description", "reminders", "use_default_reminders")
    ):
        return ToolResult(ok=False, error="nothing to change (need start, end, summary, location, description or reminders)")
    event = await _google(ctx).calendar_patch(
        event_id,
        summary=args.get("summary") or None,
        start=args.get("start") or None,
        end=args.get("end") or None,
        location=args.get("location") or None,
        description=args.get("description") or None,
        attendees=args.get("attendees") or None,
        reminders=args.get("reminders") or None,
        use_default_reminders=_bool_or_none(args.get("use_default_reminders")),
    )
    return ToolResult(
        ok=True,
        summary=f"Updated '{event.get('summary') or label}' → {_when(event)} (id {event.get('id')}).",
        display={"kind": "event_updated", "item": _card(event)},
    )


def _delete_prompt_ctx(args: dict[str, Any]) -> str:
    return _delete_prompt(args)


TOOLS: list[Tool] = [
    Tool(
        name="calendar_list_events",
        description=(
            "List calendar events in a time window (defaults to the next 7 days). "
            "Use ISO-8601 timestamps with offset for time_min/time_max, e.g. "
            "'2026-03-04T00:00:00+01:00'. Call get_current_time first if you need today's date."
        ),
        parameters={
            "type": "object",
            "properties": {
                "time_min": {"type": "string", "description": "Start of window (RFC3339). Default: now."},
                "time_max": {"type": "string", "description": "End of window (RFC3339). Default: 7 days out."},
                "query": {"type": "string", "description": "Optional free-text search on event titles."},
                "max_results": {"type": "integer", "description": "Max events (default 15)."},
            },
        },
        handler=_list,
        category="calendar",
        available_for=_ready,
        unavailable_hint="Connect a Google account (scripts/google_auth.py) to use Calendar.",
    ),
    Tool(
        name="calendar_search_events",
        description=(
            "Search the user's calendar by title/location text. Returns matching events with ids, "
            "over a window of about a month back and two months ahead."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to find in the title or location."},
                "time_min": {"type": "string", "description": "ISO-8601 window start (optional)."},
                "time_max": {"type": "string", "description": "ISO-8601 window end (optional)."},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
        handler=_search,
        category="calendar",
        available_for=_ready,
        unavailable_hint="Connect a Google account to use Calendar.",
    ),
    Tool(
        name="calendar_create_event",
        description=(
            "Create a calendar event. Confirmation is requested from the user before it is "
            "written. Times must be RFC3339 with offset, e.g. '2026-03-04T15:00:00+01:00'; "
            "end must be after start. Attendees are optional and you must NEVER ask for an email "
            "address: when the user names a person without giving an address ('a meeting with my "
            "boss'), fold the name into the title ('Meeting with boss') and create the event "
            "immediately — invitees can be added later in Google Calendar."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Event title."},
                "start": {"type": "string", "description": "Start timestamp (RFC3339)."},
                "end": {"type": "string", "description": "End timestamp (RFC3339)."},
                "description": {"type": "string"},
                "location": {"type": "string"},
                "attendees": {"type": "array", "items": {"type": "string"}, "description": "Attendee emails."},
                "reminders": {
                    "type": "array",
                    "description": "Reminders, e.g. [{'method':'popup','minutes':10}] — replaces all existing ones.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string", "enum": ["popup", "email"]},
                            "minutes": {"type": "integer", "description": "Minutes before start."},
                        },
                        "required": ["method", "minutes"],
                    },
                },
                "use_default_reminders": {"type": "boolean", "description": "True = restore calendar default reminders."},
            },
            "required": ["summary", "start", "end"],
        },
        handler=_create,
        category="calendar",
        requires_confirmation=True,
        confirmation_prompt=_create_prompt,
        preview=_create_preview,
        available_for=_ready,
        unavailable_hint="Connect a Google account to use Calendar.",
    ),
    Tool(
        name="calendar_update_event",
        description=(
            "Change an existing calendar event: move it (start/end), rename it, or edit its location. "
            "Use the id from memory when the user says 'that event' / 'move it to four'; you can also "
            "pass a 'hint' (part of the title) and the system will find the event. Confirmation is "
            "requested before the change is written. Attendees are optional — never ask for an email "
            "address unless the user wants to invite someone."
        ),
        parameters={
            "type": "object",
            "properties": {
                "event_id": {"type": "string", "description": "Event id, if you know it."},
                "hint": {"type": "string", "description": "Part of the title, e.g. 'design sync' or 'the dentist'."},
                "summary": {"type": "string", "description": "New title (optional)."},
                "start": {"type": "string", "description": "New start (RFC3339, e.g. 2026-11-25T16:00:00-08:00)."},
                "end": {"type": "string", "description": "New end (RFC3339)."},
                "location": {"type": "string"},
                "description": {"type": "string"},
                "attendees": {"type": "array", "items": {"type": "string"}},
                "reminders": {
                    "type": "array",
                    "description": "New reminders, e.g. [{'method':'popup','minutes':10}] — replaces all existing ones.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string", "enum": ["popup", "email"]},
                            "minutes": {"type": "integer", "description": "Minutes before start."},
                        },
                        "required": ["method", "minutes"],
                    },
                },
                "use_default_reminders": {"type": "boolean", "description": "True = restore calendar default reminders."},
            },
        },
        handler=_update,
        category="calendar",
        requires_confirmation=True,
        confirmation_prompt=_update_prompt,
        preview=_update_preview,
        available_for=_ready,
        unavailable_hint="Connect a Google account to use Calendar.",
    ),
    Tool(
        name="calendar_delete_event",
        description=(
            "Delete a calendar event by id (get the id from calendar_list_events). The user "
            "is asked to confirm before anything is deleted."
        ),
        parameters={
            "type": "object",
            "properties": {
                "event_id": {"type": "string", "description": "Calendar event id (from memory or a listing)."},
                "hint": {"type": "string", "description": "Part of the title if the id is unknown."},
                "summary": {"type": "string", "description": "Event title, used in the confirmation question."},
            },
        },
        handler=_delete,
        category="calendar",
        requires_confirmation=True,
        confirmation_prompt=_delete_prompt,
        preview=_delete_preview,
        danger=True,
        available_for=_ready,
        unavailable_hint="Connect a Google account to use Calendar.",
    ),
]
