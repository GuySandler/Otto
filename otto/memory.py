"""Conversation memory for reference resolution.

Voice users say "that event", "the second email", "it". The tool results that
contain the ids scroll out of the model's context as the conversation grows, so
every id-bearing result is also recorded here and the freshest handful are
injected into the system prompt as a compact block.

That is what makes "move that to four" work three turns after "what's on my
calendar?".
"""
from __future__ import annotations

import difflib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

KIND_EVENT = "event"
KIND_EMAIL = "email"
KIND_SLACK = "slack"
KIND_DRAFT = "draft"

_FUZZY_MIN = 0.55


@dataclass(slots=True)
class Entity:
    kind: str
    id: str
    label: str
    detail: str = ""
    at: float = field(default_factory=time.time)

    def line(self) -> str:
        detail = f" — {self.detail}" if self.detail else ""
        return f"- {self.kind} id={self.id} \"{self.label}\"{detail}"


def _short(value: Any, limit: int = 90) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class EntityMemory:
    def __init__(self, limit: int = 60) -> None:
        self.items: deque[Entity] = deque(maxlen=limit)

    # ------------------------------------------------------------------ write
    def add(self, kind: str, entity_id: str, label: str, detail: str = "") -> Entity | None:
        entity_id = str(entity_id or "").strip()
        if not entity_id:
            return None
        # keep the newest copy, drop stale duplicates
        self.items = deque((e for e in self.items if not (e.kind == kind and e.id == entity_id)), maxlen=self.items.maxlen)
        entity = Entity(kind=kind, id=entity_id, label=_short(label, 70), detail=_short(detail, 100))
        self.items.append(entity)
        return entity

    def remember_tool_display(self, display: dict[str, Any] | None) -> int:
        """Record the ids/entities a tool handed back to the UI."""
        if not isinstance(display, dict):
            return 0
        kind = str(display.get("kind") or "")
        added = 0

        if kind == "email_list":
            for item in display.get("items") or []:
                if self.add(KIND_EMAIL, item.get("id"), item.get("subject") or "(no subject)",
                            f"from {item.get('from') or '?'} {item.get('date') or ''}".strip()):
                    added += 1
        elif kind == "email":
            item = display.get("item") or {}
            if self.add(KIND_EMAIL, item.get("id"), item.get("subject") or "(no subject)",
                        f"from {item.get('from') or '?'} {item.get('date') or ''}".strip()):
                added += 1
        elif kind == "email_draft":
            if self.add(KIND_DRAFT, display.get("draftId"), display.get("subject") or "draft",
                        f"to {', '.join(display.get('to') or [])}".strip()):
                added += 1
        elif kind == "event_list":
            for item in display.get("items") or []:
                if self.add(KIND_EVENT, item.get("id"), item.get("title") or "(no title)",
                            _event_when(item)):
                    added += 1
        elif kind in ("event_created", "event"):
            item = display.get("item") or {}
            if self.add(KIND_EVENT, item.get("id"), item.get("title") or "(no title)", _event_when(item)):
                added += 1
        elif kind == "slack_messages":
            channel = display.get("channel") or ""
            for item in display.get("items") or []:
                if self.add(KIND_SLACK, f"{channel}:{item.get('ts')}",
                            f"#{channel}: {_short(item.get('text'), 60)}", item.get("user") or ""):
                    added += 1
        return added

    # ------------------------------------------------------------------- read
    def of_kind(self, kind: str) -> list[Entity]:
        return [item for item in self.items if item.kind == kind]

    def latest(self, kind: str | None = None) -> Entity | None:
        for item in reversed(self.items):
            if kind is None or item.kind == kind:
                return item
        return None

    def resolve(self, kind: str | None, hint: str | None = None) -> Entity | None:
        """Find the entity the user means. No hint → most recent of that kind."""
        candidates = self.of_kind(kind) if kind else list(self.items)
        if not candidates:
            return None
        hint = str(hint or "").strip().strip("\"'")
        if not hint:
            return candidates[-1]
        lowered = hint.lower()
        # exact-ish first, then fuzzy, then any word overlap
        for item in reversed(candidates):
            if lowered == item.label.lower() or lowered == item.id.lower():
                return item
        scored: Iterable[tuple[float, Entity]] = sorted(
            ((difflib.SequenceMatcher(None, lowered, item.label.lower()).ratio(), item) for item in candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )
        for score, item in scored:
            if score >= _FUZZY_MIN:
                return item
        words = [w for w in lowered.split() if len(w) > 3]
        for item in reversed(candidates):
            if any(word in item.label.lower() for word in words):
                return item
        return None

    # ------------------------------------------------------------------ prompt
    def prompt_block(self, *, max_lines: int = 10) -> str:
        if not self.items:
            return ""
        # newest first, but keep the calendar events and emails visible
        events = self.of_kind(KIND_EVENT)[-4:]
        emails = self.of_kind(KIND_EMAIL)[-4:]
        drafts = self.of_kind(KIND_DRAFT)[-2:]
        slack = self.of_kind(KIND_SLACK)[-2:]
        chosen = events + emails + drafts + slack
        if not chosen:
            chosen = list(self.items)[-max_lines:]
        lines = [entity.line() for entity in chosen[-max_lines:]]
        return (
            "Items seen earlier in this conversation (ids are real — use them when the user "
            "refers back with 'that', 'it', 'the first one'):\n" + "\n".join(lines)
        )


def _event_when(item: dict[str, Any]) -> str:
    start = str(item.get("start") or "")
    end = str(item.get("end") or "")
    if item.get("all_day"):
        return f"all day {start[:10]}"
    if "T" in start:
        when = start[:16].replace("T", " ")
        if "T" in end:
            when += f"–{end[11:16]}"
        return when
    return start
