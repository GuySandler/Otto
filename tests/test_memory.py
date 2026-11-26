"""Cross-turn reference memory: "move *that* to four"."""
from __future__ import annotations

from otto.memory import KIND_DRAFT, KIND_EMAIL, KIND_EVENT, KIND_SLACK, EntityMemory


def test_remembers_events_from_tool_display():
    memory = EntityMemory()
    added = memory.remember_tool_display(
        {
            "kind": "event_list",
            "items": [
                {"id": "evt_1", "title": "Design sync", "start": "2026-11-25T10:00:00-08:00",
                 "end": "2026-11-25T10:30:00-08:00"},
                {"id": "evt_2", "title": "Dentist", "start": "2026-11-26T15:00:00-08:00",
                 "end": "2026-11-26T16:00:00-08:00", "all_day": False},
            ],
        }
    )
    assert added == 2
    entity = memory.resolve(KIND_EVENT, None)
    assert entity is not None and entity.id == "evt_2"  # newest wins when ambiguous
    assert "11-26 15:00" in entity.detail


def test_resolves_by_fuzzy_title_and_by_id():
    memory = EntityMemory()
    memory.remember_tool_display(
        {"kind": "event_list", "items": [
            {"id": "evt_1", "title": "Design sync", "start": "2026-11-25T10:00:00-08:00", "end": ""},
            {"id": "evt_2", "title": "Dentist appointment", "start": "2026-11-26T15:00:00-08:00", "end": ""},
        ]}
    )
    assert memory.resolve(KIND_EVENT, "the dentist one").id == "evt_2"
    assert memory.resolve(KIND_EVENT, "design").id == "evt_1"
    assert memory.resolve(KIND_EVENT, "evt_1").id == "evt_1"
    assert memory.resolve(KIND_EVENT, "the quarterly tax filing") is None


def test_remembers_emails_drafts_and_slack():
    memory = EntityMemory()
    memory.remember_tool_display(
        {"kind": "email_list", "items": [
            {"id": "m1", "subject": "Lease renewal", "from": "sarah@x.com", "date": "yesterday"},
        ]}
    )
    memory.remember_tool_display(
        {"kind": "email_draft", "draftId": "d1", "to": ["sarah@x.com"], "subject": "Re: Lease"}
    )
    memory.remember_tool_display(
        {"kind": "slack_messages", "channel": "design", "items": [
            {"ts": "111.2", "user": "sam", "text": "ship it"},
        ]}
    )
    assert memory.resolve(KIND_EMAIL, "lease").id == "m1"
    assert memory.resolve(KIND_DRAFT).id == "d1"
    assert memory.resolve(KIND_SLACK, "ship it").id == "design:111.2"


def test_duplicate_ids_are_deduplicated_newest_first():
    memory = EntityMemory()
    for when in ("2026-01-01T09:00:00", "2026-01-02T09:00:00"):
        memory.remember_tool_display(
            {"kind": "event_list", "items": [{"id": "evt_1", "title": "Standup", "start": when, "end": ""}]}
        )
    assert len(memory.of_kind(KIND_EVENT)) == 1
    assert "01-02" in memory.resolve(KIND_EVENT).detail


def test_prompt_block_is_compact_and_mentions_ids():
    memory = EntityMemory()
    memory.remember_tool_display(
        {"kind": "event_list", "items": [{"id": "evt_1", "title": "Design sync", "start": "2026-11-25T10:00:00", "end": ""}]}
    )
    memory.remember_tool_display(
        {"kind": "email_list", "items": [{"id": "m1", "subject": "Lease renewal", "from": "sarah@x.com", "date": "yesterday"}]}
    )
    block = memory.prompt_block()
    assert "evt_1" in block and "m1" in block
    assert "Design sync" in block and "Lease renewal" in block
    assert len(block.splitlines()) <= 11


def test_empty_memory_yields_empty_block():
    assert EntityMemory().prompt_block() == ""


def test_ignores_unknown_display_kinds():
    memory = EntityMemory()
    assert memory.remember_tool_display({"kind": "text", "text": "hi"}) == 0
    assert memory.remember_tool_display(None) == 0
