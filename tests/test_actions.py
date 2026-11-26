"""Actions: things Otto made that the human has to finish (drafts, and the API around them)."""
from __future__ import annotations

import time

import pytest

from otto.actions import (
    DONE,
    OPEN,
    ActionStore,
    actions_from_display,
    gmail_url,
)
from otto.db import Database
from otto.users import UserStore


@pytest.fixture
def stores(tmp_path):
    db = Database(tmp_path / "otto.db")
    return UserStore(db), ActionStore(db), db


def test_gmail_deep_links():
    assert gmail_url(draft_id="d1", account_email="me@x.com") == (
        "https://mail.google.com/mail/u/me%40x.com/#drafts?compose=d1"
    )
    assert gmail_url(draft_id="d1") == "https://mail.google.com/mail/u/0/#drafts?compose=d1"
    assert gmail_url(thread_id="t9") == "https://mail.google.com/mail/u/0/#all/t9"
    assert gmail_url() == "https://mail.google.com/mail/u/0/#drafts"


def test_actions_from_a_draft_display():
    specs = actions_from_display(
        {
            "kind": "email_draft",
            "draftId": "draft_42",
            "to": ["sarah@acme.com", "sam@acme.com"],
            "subject": "Re: lease renewal",
            "body": "Tuesday works for me.",
        },
        account_email="me@x.com",
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec["kind"] == "draft"
    assert spec["title"] == "Re: lease renewal"
    assert spec["subtitle"] == "to sarah@acme.com, sam@acme.com"
    assert "compose=draft_42" in spec["url"]
    assert spec["ref"] == "draft:draft_42"


def test_actions_from_unknown_or_empty_displays():
    assert actions_from_display(None) == []
    assert actions_from_display({"kind": "email_list", "items": []}) == []
    assert actions_from_display({"kind": "text", "text": "hi"}) == []


def test_created_events_are_recorded_but_already_done():
    specs = actions_from_display({"kind": "event_created", "item": {"id": "e1", "title": "Design sync"}})
    assert specs[0]["kind"] == "event"
    assert specs[0]["status_override"] == DONE


def test_record_list_and_status(stores):
    users, actions, _db = stores
    user = users.create_user("a@b.co", "longenough1")

    action = actions.record(user.id, kind="draft", title="Re: lease", subtitle="to sarah@acme.com", ref="draft:d1")
    assert action.status == OPEN
    assert actions.count(user.id) == 1

    listed = actions.list(user.id)
    assert [item.id for item in listed] == [action.id]
    assert listed[0].subtitle == "to sarah@acme.com"

    finished = actions.set_status(user.id, action.id, DONE)
    assert finished is not None and finished.status == DONE
    assert actions.list(user.id) == []
    assert [item.id for item in actions.list(user.id, status=DONE)] == [action.id]
    assert actions.count(user.id) == 0


def test_same_draft_updates_instead_of_duplicating(stores):
    users, actions, _db = stores
    user = users.create_user("a@b.co", "longenough1")

    first = actions.record(user.id, kind="draft", title="Lease", ref="draft:d1", url="u1")
    second = actions.record(user.id, kind="draft", title="Lease (edited)", ref="draft:d1", url="u2")
    assert first.id == second.id
    assert second.title == "Lease (edited)"
    assert second.url == "u2"
    assert actions.count(user.id) == 1

    # once finished, a new draft with the same ref is a new action
    actions.set_status(user.id, first.id, DONE)
    third = actions.record(user.id, kind="draft", title="Lease", ref="draft:d1")
    assert third.id != first.id


def test_actions_are_per_user(stores):
    users, actions, _db = stores
    a = users.create_user("a@b.co", "longenough1")
    b = users.create_user("c@d.co", "longenough1")
    actions.record(a.id, kind="draft", title="mine")
    assert actions.count(a.id) == 1
    assert actions.count(b.id) == 0
    assert actions.list(b.id) == []


def test_status_must_be_known(stores):
    users, actions, _db = stores
    user = users.create_user("a@b.co", "longenough1")
    action = actions.record(user.id, kind="draft", title="x")
    with pytest.raises(ValueError):
        actions.set_status(user.id, action.id, "archived")


def test_record_requires_a_user(stores):
    _users, actions, _db = stores
    with pytest.raises(ValueError):
        actions.record("", kind="draft", title="x")


def test_list_is_newest_first(stores):
    users, actions, _db = stores
    user = users.create_user("a@b.co", "longenough1")
    for index in range(3):
        actions.record(user.id, kind="draft", title=f"draft {index}")
        time.sleep(0.001)
    titles = [item.title for item in actions.list(user.id)]
    assert titles == ["draft 2", "draft 1", "draft 0"]


def test_clear_removes_everything(stores):
    users, actions, _db = stores
    user = users.create_user("a@b.co", "longenough1")
    actions.record(user.id, kind="draft", title="x")
    actions.clear(user.id)
    assert actions.count(user.id) == 0
