"""Things Otto produced that need the human.

A voice assistant can't press send, and a conversation ends. So anything a tool
creates *for* the user — today an email draft, tomorrow whatever else — is
persisted as an **action** with a deep link, and the dashboard surfaces the open
ones. This is the hand-off between "Otto did the work" and "you finish it".
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

log = logging.getLogger("otto.actions")

OPEN = "open"
DONE = "done"
DISMISSED = "dismissed"
STATUSES = {OPEN, DONE, DISMISSED}

KIND_DRAFT = "draft"
KIND_EMAIL = "email"
KIND_EVENT = "event"

MAIL_BASE = "https://mail.google.com/mail/u"


@dataclass(slots=True)
class Action:
    id: str
    user_id: str
    kind: str
    title: str
    subtitle: str = ""
    detail: str = ""
    url: str = ""
    status: str = OPEN
    ref: str = ""
    source: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "subtitle": self.subtitle,
            "detail": self.detail,
            "url": self.url,
            "status": self.status,
            "source": self.source,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }


# --------------------------------------------------------------------- links
def gmail_url(*, draft_id: str = "", message_id: str = "", thread_id: str = "", account_email: str = "") -> str:
    """Best-effort deep link into the user's Gmail.

    Gmail's per-draft compose deep links are unreliable with API draft ids, so
    drafts land on the Drafts label — one click from the message itself. A
    message or thread id still deep-links to the message.
    """
    user = f"/{quote(account_email)}" if account_email else "/0"
    if draft_id:
        return f"{MAIL_BASE}{user}/#drafts"
    target = message_id or thread_id
    if target:
        return f"{MAIL_BASE}{user}/#all/{quote(str(target))}"
    return f"{MAIL_BASE}{user}/#drafts"


def actions_from_display(display: dict[str, Any] | None, *, account_email: str = "") -> list[dict[str, Any]]:
    """Turn a tool's UI payload into zero or more persisted actions."""
    if not isinstance(display, dict):
        return []
    kind = str(display.get("kind") or "")
    if kind == "email_draft":
        draft_id = str(display.get("draftId") or "")
        to = display.get("to") or []
        recipients = ", ".join(str(r) for r in to) if isinstance(to, list) else str(to)
        return [
            {
                "kind": KIND_DRAFT,
                "title": str(display.get("subject") or "(no subject)"),
                "subtitle": f"to {recipients}" if recipients else "draft",
                "detail": str(display.get("body") or "")[:600],
                "url": gmail_url(draft_id=draft_id, account_email=account_email),
                "ref": f"draft:{draft_id}" if draft_id else "",
                "detail_ref": draft_id,
            }
        ]
    if kind in ("event_created", "event_updated"):
        item = display.get("item") or {}
        return [
            {
                "kind": KIND_EVENT,
                "title": str(item.get("title") or "calendar event"),
                "subtitle": str(item.get("start") or ""),
                "detail": str(item.get("location") or ""),
                "url": str(item.get("link") or "https://calendar.google.com/calendar/u/0/r"),
                "ref": f"event:{item.get('id')}" if item.get("id") else "",
                "status_override": DONE,  # the calendar already has it; nothing to finish
            }
        ]
    return []


# --------------------------------------------------------------------- store
class ActionStore:
    def __init__(self, db) -> None:
        self.db = db

    # ------------------------------------------------------------------ writes
    def record(
        self,
        user_id: str,
        *,
        kind: str,
        title: str,
        subtitle: str = "",
        detail: str = "",
        url: str = "",
        ref: str = "",
        source: str = "",
        status: str = OPEN,
        payload: dict[str, Any] | None = None,
    ) -> Action:
        if not user_id:
            raise ValueError("an action needs a user")
        now = time.time()
        if ref:
            existing = self.db.query_one(
                "SELECT id, created_at FROM actions WHERE user_id = ? AND ref = ? AND status = ?",
                (user_id, ref, OPEN),
            )
            if existing is not None:
                self.db.execute(
                    "UPDATE actions SET title = ?, subtitle = ?, detail = ?, url = ?, source = ?, "
                    "updated_at = ?, payload = ? WHERE id = ?",
                    (title, subtitle, detail, url, source, now, json.dumps(payload or {}), existing["id"]),
                )
                found = self.get(user_id, str(existing["id"]))
                assert found is not None
                return found
        action_id = uuid.uuid4().hex[:12]
        self.db.execute(
            "INSERT INTO actions (id, user_id, kind, title, subtitle, detail, url, status, ref, source, "
            "created_at, updated_at, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                action_id,
                user_id,
                kind,
                title[:200],
                subtitle[:200],
                detail[:1000],
                url,
                status if status in STATUSES else OPEN,
                ref,
                source,
                now,
                now,
                json.dumps(payload or {}),
            ),
        )
        action = self.get(user_id, action_id)
        assert action is not None
        log.info("action recorded: %s (%s) for %s", action.title[:60], action.kind, user_id[:8])
        return action

    def set_status(self, user_id: str, action_id: str, status: str) -> Action | None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        self.db.execute(
            "UPDATE actions SET status = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (status, time.time(), action_id, user_id),
        )
        return self.get(user_id, action_id)

    def clear(self, user_id: str) -> None:
        self.db.execute("DELETE FROM actions WHERE user_id = ?", (user_id,))

    # ------------------------------------------------------------------- reads
    def get(self, user_id: str, action_id: str) -> Action | None:
        row = self.db.query_one("SELECT * FROM actions WHERE id = ? AND user_id = ?", (action_id, user_id))
        return self._row(row) if row else None

    def list(self, user_id: str, *, status: str = OPEN, limit: int = 50) -> list[Action]:
        if status == "all":
            rows = self.db.query(
                "SELECT * FROM actions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit)
            )
        else:
            rows = self.db.query(
                "SELECT * FROM actions WHERE user_id = ? AND status = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, status, limit),
            )
        return [self._row(row) for row in rows]

    def count(self, user_id: str, *, status: str = OPEN) -> int:
        row = self.db.query_one(
            "SELECT COUNT(*) AS n FROM actions WHERE user_id = ? AND status = ?", (user_id, status)
        )
        return int(row["n"]) if row else 0

    @staticmethod
    def _row(row: Any) -> Action:
        try:
            payload = json.loads(row["payload"] or "{}")
        except ValueError:
            payload = {}
        return Action(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            kind=str(row["kind"]),
            title=str(row["title"]),
            subtitle=str(row["subtitle"] or ""),
            detail=str(row["detail"] or ""),
            url=str(row["url"] or ""),
            status=str(row["status"]),
            ref=str(row["ref"] or ""),
            source=str(row["source"] or ""),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            payload=payload,
        )
