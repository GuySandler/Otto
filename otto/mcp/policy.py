"""How an MCP tool becomes an Otto tool.

The MCP server publishes the schema and the risk annotations; this module decides
the *product* around it:

* an Otto-side name (so personas, the entity memory and the UI keep working),
* whether the user must confirm before it runs, and the sentence we speak,
* how the result is summarised for speech and rendered as a card.

The summarisers are deliberately tolerant about key names: Google's MCP servers
return protobuf-ish JSON (`plaintextBody`, `toRecipients`, `start.dateTime`) while
other servers use snake_case, and a summary that degrades to compact text is much
better than a crash mid-conversation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .client import MCPCallResult


# --------------------------------------------------------------------- helpers
def _first(payload: Any, *keys: str) -> list[dict[str, Any]]:
    """First list-of-dicts under any of ``keys`` (or the payload itself)."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    # one level deeper, e.g. {"result": {"messages": [...]}}
    for value in payload.values():
        if isinstance(value, dict):
            found = _first(value, *keys)
            if found:
                return found
    return []


def _pick(item: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            return str(value).strip()
        if isinstance(value, dict):
            inner = value.get("dateTime") or value.get("date") or value.get("email") or value.get("name")
            if inner:
                return str(inner)
        if isinstance(value, list) and value:
            parts = [str(v.get("email") or v.get("address") or v) if isinstance(v, dict) else str(v) for v in value]
            joined = ", ".join(p for p in parts if p)[:120]
            if joined:
                return joined
    return default


def _short(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _when(event: dict[str, Any]) -> str:
    start = event.get("start")
    end = event.get("end")
    start_text = ""
    end_text = ""
    if isinstance(start, dict):
        start_text = str(start.get("dateTime") or start.get("date") or "")
    elif isinstance(start, str):
        start_text = start
    if isinstance(end, dict):
        end_text = str(end.get("dateTime") or end.get("date") or "")
    elif isinstance(end, str):
        end_text = end
    if not start_text:
        start_text = _pick(event, "startTime", "start_time", "startsAt")
    if start_text and "T" in start_text:
        when = start_text[:16].replace("T", " ")
        if "T" in end_text:
            when += f"–{end_text[11:16]}"
        return when
    return start_text


def _email_card(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _pick(item, "id", "messageId", "message_id", "threadId"),
        "from": _pick(item, "from", "sender", "fromRecipients", "from_"),
        "to": _pick(item, "to", "toRecipients", "to_"),
        "subject": _pick(item, "subject", "title"),
        "date": _pick(item, "date", "internalDate", "receivedAt"),
        "snippet": _short(_pick(item, "snippet", "body", "plaintextBody", "text"), 180),
        "unread": bool(item.get("unread") or item.get("isUnread")),
    }


def _event_card(item: dict[str, Any]) -> dict[str, Any]:
    attendees = item.get("attendees")
    return {
        "id": _pick(item, "id", "eventId", "event_id"),
        "title": _pick(item, "summary", "title", "name") or "(no title)",
        "start": _pick(item, "start", "startTime", "start_time"),
        "end": _pick(item, "end", "endTime", "end_time"),
        "all_day": bool(item.get("allDay")) or isinstance(item.get("start"), dict) and bool((item.get("start") or {}).get("date")),
        "location": _pick(item, "location", "place"),
        "attendees": [
            {"email": _pick(a, "email", "address"), "displayName": _pick(a, "displayName", "name")}
            for a in attendees or []
            if isinstance(a, dict)
        ],
        "link": _pick(item, "htmlLink", "html_link", "link"),
    }


def _generic(result: MCPCallResult) -> tuple[str, dict[str, Any] | None]:
    payload = result.json_payload()
    text = result.text
    if payload is not None:
        summary = json.dumps(payload, ensure_ascii=False)[:2000]
    else:
        summary = text[:2000]
    if not summary:
        summary = "(the tool returned nothing)"
    return summary, {"kind": "mcp", "text": _short(text or summary, 600)}


# ---------------------------------------------------------------------- policy
@dataclass(slots=True)
class ToolPolicy:
    alias: str
    category: str
    requires_confirmation: bool | None = None   # None → derive from MCP annotations
    danger: bool = False
    #: appended to the tool description the model sees — voice-model guardrails
    note: str = ""
    prompt: Callable[[dict[str, Any]], str] | None = None
    preview: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    summarize: Callable[[MCPCallResult, dict[str, Any]], tuple[str, dict[str, Any] | None]] = _generic


# ------------------------------------------------------------------- gmail side
def _thread_list(result: MCPCallResult, args: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    payload = result.json_payload()
    items = _first(payload, "threads", "messages", "items", "results")
    if not items:
        return "No mail matched that search.", {"kind": "email_list", "query": args.get("query", ""), "items": []}
    lines = []
    cards = []
    for item in items[:12]:
        card = _email_card(item)
        cards.append(card)
        unread = " (unread)" if card["unread"] else ""
        lines.append(
            f"- id={card['id']} | from {card['from'] or '?'} | {card['subject'] or '(no subject)'} | "
            f"{card['date']}{unread} | {_short(card['snippet'], 100)}"
        )
    summary = f"{len(items)} result(s):\n" + "\n".join(lines)
    return summary, {"kind": "email_list", "query": args.get("query", ""), "items": cards}


def _thread_read(result: MCPCallResult, args: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    payload = result.json_payload()
    envelope = payload if isinstance(payload, dict) else {}
    messages = _first(payload, "messages", "message", "thread")
    body_source = messages[0] if messages else envelope
    body = _pick(body_source, "plaintextBody", "body", "snippet", "text")
    if not body and messages:
        body = "\n\n".join(
            f"{_pick(m, 'from', 'sender')}: {_short(_pick(m, 'plaintextBody', 'body', 'snippet'), 600)}"
            for m in messages[:6]
        )
    card = _email_card(body_source)
    card["body"] = _short(body, 4000)
    header = (
        f"From: {card['from'] or '?'}\nTo: {card['to'] or '?'}\nSubject: {card['subject'] or '(no subject)'}\n"
        f"Date: {card['date'] or '?'}\n\n{_short(body, 1600)}"
    )
    return header, {"kind": "email", "item": card}


def _draft_created(result: MCPCallResult, args: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    payload = result.json_payload() or {}
    draft = payload.get("draft") if isinstance(payload.get("draft"), dict) else payload
    draft_id = _pick(draft, "id", "draftId", "draft_id")
    to = _pick(draft, "toRecipients", "to") or _pick(args, "to")
    subject = _pick(draft, "subject") or _pick(args, "subject")
    body = _pick(draft, "plaintextBody", "body") or _pick(args, "body")
    summary = f"Draft created (id {draft_id or '?'}) to {to or '?'} with subject '{subject or '(no subject)'}'."
    return summary, {
        "kind": "email_draft",
        "draftId": draft_id,
        "to": [part.strip() for part in str(to or "").split(",") if part.strip()],
        "subject": subject,
        "body": _short(body, 2000),
    }


def _labels(result: MCPCallResult, args: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    items = _first(result.json_payload(), "labels", "items")
    names = [_pick(item, "name", "displayName", "id") for item in items][:40]
    return ("Labels: " + ", ".join(n for n in names if n)) if names else "No labels.", {
        "kind": "list",
        "items": [n for n in names if n],
    }


def _label_change(verb: str) -> Callable[[MCPCallResult, dict[str, Any]], tuple[str, dict[str, Any] | None]]:
    def summarize(result: MCPCallResult, args: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        if result.is_error:
            return f"{verb} failed: {_short(result.text, 200)}", None
        target = _pick(args, "messageId", "threadId", "id")
        return f"{verb} {target or 'the message'}.", {"kind": "text", "text": f"{verb} {target}"}

    return summarize


# ---------------------------------------------------------------- calendar side
def _event_list(result: MCPCallResult, args: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    items = _first(result.json_payload(), "events", "items", "results")
    if not items:
        requested = any(
            args.get(key) for key in ("startTime", "endTime", "timeMin", "timeMax", "time_min", "time_max")
        )
        if requested:
            return "No events in that window.", {"kind": "event_list", "items": []}
        # no range given: the server answers for its own default window, which is
        # not the same as "the calendar is empty" — say so, so the model asks.
        return (
            "No events in the server's default window (no startTime/endTime was passed). "
            "Ask the user which range they want, or call again with explicit startTime and endTime.",
            {"kind": "event_list", "items": []},
        )
    cards = [_event_card(item) for item in items[:20]]
    lines = []
    for card in cards:
        extras = []
        if card["location"]:
            extras.append(str(card["location"]))
        if card["attendees"]:
            extras.append(f"{len(card['attendees'])} attendees")
        suffix = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"- id={card['id']} | {card['title']} | {card['start']}{suffix}")
    return f"{len(cards)} event(s):\n" + "\n".join(lines), {"kind": "event_list", "items": cards}


def _event_one(kind: str) -> Callable[[MCPCallResult, dict[str, Any]], tuple[str, dict[str, Any] | None]]:
    def summarize(result: MCPCallResult, args: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        payload = result.json_payload()
        envelope = payload if isinstance(payload, dict) else {}
        event = envelope.get("event") if isinstance(envelope.get("event"), dict) else envelope
        if not event:
            return _generic(result, args)
        card = _event_card(event)
        verb = {"event_created": "Created", "event_updated": "Updated", "event_deleted": "Deleted"}.get(kind, "Event")
        return f"{verb} '{card['title']}' — {card['start']} (id {card['id']}).", {"kind": kind, "item": card}

    return summarize


def _calendars(result: MCPCallResult, args: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    items = _first(result.json_payload(), "calendars", "items")
    names = [_pick(item, "summary", "name", "id") for item in items][:20]
    return ("Calendars: " + ", ".join(n for n in names if n)) if names else "No calendars.", {
        "kind": "list",
        "items": [n for n in names if n],
    }


def _free_slots(result: MCPCallResult, args: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    items = _first(result.json_payload(), "slots", "suggestions", "items", "timeSlots")
    labels = [_when(item) or _pick(item, "start", "startTime") for item in items][:8]
    labels = [label for label in labels if label]
    if not labels:
        return _generic(result, args)
    return "Suggested: " + "; ".join(labels), {"kind": "list", "items": labels}


def _event_create_prompt(args: dict[str, Any]) -> str:
    title = _pick(args, "summary", "title") or "an event"
    when = _when(args) or "that time"
    return f"I'll put {title} on the calendar for {when}. Should I add it?"


def _short_when(value: str) -> str:
    """'2026-11-26T15:00:00-08:00' → '2026-11-26 at 15:00' — speakable."""
    text = str(value)
    return text[:16].replace("T", " at ") if "T" in text else text


def _describe_event_change(args: dict[str, Any]) -> list[str]:
    """Human words for what an update_event call will actually change."""
    bits: list[str] = []
    start = _pick(args, "start", "startTime")
    end = _pick(args, "end", "endTime")
    if start:
        bits.append(f"move it to {_short_when(start)}")
    if end:
        bits.append(f"end at {_short_when(end)}")
    if args.get("location"):
        bits.append(f"set the location to {args['location']}")
    reminders = args.get("overrideReminders") or args.get("reminders")
    if isinstance(reminders, list) and reminders:
        parts = []
        for item in reminders:
            if isinstance(item, dict):
                method = str(item.get("method") or "popup")
                article = "an" if method == "email" else "a"
                parts.append(f"{article} {method} reminder {item.get('minutes')} minutes before")
        if parts:
            bits.append("set " + " and ".join(parts))
    if args.get("useDefaultReminders"):
        bits.append("restore the calendar's default reminders")
    if isinstance(args.get("description"), str) and args["description"].strip():
        bits.append("rewrite its description")
    return bits


def _event_update_prompt(args: dict[str, Any]) -> str:
    # the id (if any) is swapped for the remembered title by the session
    target = _pick(args, "eventId", "event_id", "hint") or "that event"
    bits = _describe_event_change(args)
    summary = _pick(args, "summary", "title")
    if bits:
        return f"I'll update {target}: {', '.join(bits[:3])}. Okay?"
    if summary:
        return f"I'll rename {target} to '{summary}'. Okay?"
    return f"I'll update {target}. Okay?"


def _event_delete_prompt(args: dict[str, Any]) -> str:
    title = _pick(args, "summary", "title", "eventId", "event_id") or "that event"
    return f"Do you want me to delete {title} from your calendar?"


def _event_create_preview(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "event_create",
        "title": _pick(args, "summary", "title"),
        "start": _pick(args, "start", "startTime"),
        "end": _pick(args, "end", "endTime"),
        "location": _pick(args, "location"),
        "attendees": args.get("attendees") or [],
    }


def _event_delete_preview(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "event_delete",
        "id": _pick(args, "eventId", "event_id", "id"),
        "title": _pick(args, "summary", "title"),
    }


# ------------------------------------------------------------------- the table
POLICIES: dict[tuple[str, str], ToolPolicy] = {
    # gmail -------------------------------------------------------------------
    ("gmail", "search_threads"): ToolPolicy("gmail_search", "gmail", summarize=_thread_list),
    ("gmail", "get_thread"): ToolPolicy("gmail_thread", "gmail", summarize=_thread_read),
    ("gmail", "get_message"): ToolPolicy("gmail_read", "gmail", summarize=_thread_read),
    ("gmail", "create_draft"): ToolPolicy("gmail_draft", "gmail", summarize=_draft_created),
    ("gmail", "list_drafts"): ToolPolicy("gmail_list_drafts", "gmail", summarize=_thread_list),
    ("gmail", "list_labels"): ToolPolicy("gmail_labels", "gmail", summarize=_labels),
    ("gmail", "label_message"): ToolPolicy(
        "gmail_label", "gmail", requires_confirmation=False, summarize=_label_change("Labelled")
    ),
    ("gmail", "label_thread"): ToolPolicy(
        "gmail_label_thread", "gmail", requires_confirmation=False, summarize=_label_change("Labelled")
    ),
    ("gmail", "unlabel_message"): ToolPolicy(
        "gmail_unlabel", "gmail", requires_confirmation=False, summarize=_label_change("Unlabelled")
    ),
    ("gmail", "unlabel_thread"): ToolPolicy(
        "gmail_unlabel_thread", "gmail", requires_confirmation=False, summarize=_label_change("Unlabelled")
    ),
    ("gmail", "get_draft"): ToolPolicy("gmail_read_draft", "gmail", summarize=_thread_read),
    ("gmail", "update_message_labels"): ToolPolicy(
        "gmail_update_labels", "gmail", requires_confirmation=False, summarize=_label_change("Updated labels on")
    ),
    ("gmail", "create_label"): ToolPolicy(
        "gmail_create_label", "gmail", requires_confirmation=False, summarize=_label_change("Created label")
    ),
    # trashing, spam and "sensitive" labels are the writes that need a human
    ("gmail", "trash_thread"): ToolPolicy(
        "gmail_trash_thread", "gmail", requires_confirmation=True, danger=True, summarize=_label_change("Trashed thread")
    ),
    ("gmail", "trash_message"): ToolPolicy(
        "gmail_trash", "gmail", requires_confirmation=True, danger=True, summarize=_label_change("Trashed")
    ),
    ("gmail", "mark_thread_spam"): ToolPolicy(
        "gmail_mark_spam_thread", "gmail", requires_confirmation=True, summarize=_label_change("Marked as spam")
    ),
    ("gmail", "mark_message_spam"): ToolPolicy(
        "gmail_mark_spam", "gmail", requires_confirmation=True, summarize=_label_change("Marked as spam")
    ),
    ("gmail", "apply_sensitive_thread_label"): ToolPolicy(
        "gmail_sensitive_label_thread", "gmail", requires_confirmation=True, summarize=_label_change("Labelled thread")
    ),
    ("gmail", "apply_sensitive_message_label"): ToolPolicy(
        "gmail_sensitive_label", "gmail", requires_confirmation=True, summarize=_label_change("Labelled")
    ),
    ("gmail", "untrash_thread"): ToolPolicy(
        "gmail_untrash_thread", "gmail", requires_confirmation=False, summarize=_label_change("Restored thread")
    ),
    ("gmail", "untrash_message"): ToolPolicy(
        "gmail_untrash", "gmail", requires_confirmation=False, summarize=_label_change("Restored")
    ),
    ("gmail", "unmark_thread_spam"): ToolPolicy(
        "gmail_unmark_spam_thread", "gmail", requires_confirmation=False, summarize=_label_change("Cleared spam on")
    ),
    ("gmail", "unmark_message_spam"): ToolPolicy(
        "gmail_unmark_spam", "gmail", requires_confirmation=False, summarize=_label_change("Cleared spam on")
    ),
    # calendar ----------------------------------------------------------------
    ("calendar", "list_events"): ToolPolicy("calendar_list_events", "calendar", summarize=_event_list),
    ("calendar", "search_events"): ToolPolicy("calendar_search_events", "calendar", summarize=_event_list),
    ("calendar", "get_event"): ToolPolicy("calendar_get_event", "calendar", summarize=_event_one("event")),
    ("calendar", "list_calendars"): ToolPolicy("calendar_list_calendars", "calendar", summarize=_calendars),
    ("calendar", "suggest_time"): ToolPolicy("calendar_suggest_time", "calendar", summarize=_free_slots),
    ("calendar", "create_event"): ToolPolicy(
        "calendar_create_event",
        "calendar",
        requires_confirmation=True,
        prompt=_event_create_prompt,
        preview=_event_create_preview,
        summarize=_event_one("event_created"),
        note=("Attendees are optional and you must NEVER ask for an email address: when the user names a person without giving an address ('a meeting with my boss'), fold the name into the title ('Meeting with boss') and create the event immediately — invitees can be added later in Google Calendar. If the user asks for reminders, set overrideReminders=[{'method':'popup'|'email','minutes':N}] — it replaces all existing reminders."),
    ),
    ("calendar", "update_event"): ToolPolicy(
        "calendar_update_event",
        "calendar",
        requires_confirmation=True,
        prompt=_event_update_prompt,
        preview=_event_create_preview,
        summarize=_event_one("event_updated"),
        note=(
            "Attendees are optional and you must never ask for an email address — "
            "apply the change with what you already know. "
            "If the user asks about reminders or notification times: "
            "overrideReminders=[{'method':'popup'|'email','minutes':N}] replaces all reminders, "
            "or useDefaultReminders=true restores the calendar defaults."
        ),
    ),
    ("calendar", "delete_event"): ToolPolicy(
        "calendar_delete_event",
        "calendar",
        requires_confirmation=True,
        danger=True,
        prompt=_event_delete_prompt,
        preview=_event_delete_preview,
        summarize=_event_one("event_deleted"),
    ),
    ("calendar", "respond_to_event"): ToolPolicy(
        "calendar_respond_to_event",
        "calendar",
        requires_confirmation=True,
        prompt=lambda args: f"Should I send your '{_pick(args, 'responseStatus', 'response') or 'yes'}' reply to that invitation?",
        summarize=_event_one("event_updated"),
    ),
}


def policy_for(server: str, tool: str, *, fallback_prefix: bool = True) -> ToolPolicy:
    found = POLICIES.get((server, tool))
    if found is not None:
        return found
    alias = f"{server}_{tool}" if fallback_prefix else tool
    return ToolPolicy(alias=alias, category=server)


def confirmation_for(policy: ToolPolicy, info) -> bool:
    """Writes that destroy something always ask; explicit policy wins over hints."""
    if policy.requires_confirmation is not None:
        return policy.requires_confirmation
    if info is not None and getattr(info, "destructive", False):
        return True
    return False
