"""Conversation harness: turn taking, streaming, tools, confirmations, barge-in."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
import yaml

from otto.actions import ActionStore
from otto.db import Database
from otto.integrations import Integrations
from otto.personality import PersonalityStore
from otto.pipeline.llm import StreamDone, TextDelta
from otto.protocol import Config, ToolCall
from otto.session import Session
from otto.settings import Settings
from otto.tools.base import Tool, ToolResult
from otto.tools.registry import ToolRegistry

from .fakes import Collector, FakeLLM, FakeSTT, FakeTTS

REPO = Path(__file__).resolve().parent.parent


class FakeIntegrations:
    def __init__(self) -> None:
        self.google = None
        self.slack = None

    def status(self):
        return {}

    async def aclose(self):
        return None


def make_settings(tmp_path: Path) -> Settings:
    shutil.copytree(REPO / "config", tmp_path / "config")
    app_cfg = yaml.safe_load((REPO / "config" / "app.yaml").read_text())
    # unit tests are offline: MCP is exercised separately with a fake server
    app_cfg["mcp"] = {"enabled": False}
    return Settings(root=tmp_path, lunaroute_api_key="test-key", app_cfg=app_cfg)


def make_session(
    tmp_path: Path,
    *,
    llm: FakeLLM | None = None,
    tts: FakeTTS | None = None,
    stt: FakeSTT | None = None,
    tools: list[Tool] | None = None,
    personality: str = "otto",
) -> tuple[Session, Collector]:
    settings = make_settings(tmp_path)
    registry = ToolRegistry()
    for tool in tools or []:
        registry.add(tool)
    collector = Collector()
    actions = ActionStore(Database(tmp_path / "otto-actions.db"))
    session = Session(
        session_id="test",
        settings=settings,
        llm=llm or FakeLLM(),
        stt=stt or FakeSTT(),
        tts=tts or FakeTTS(),
        registry=registry,
        personalities=PersonalityStore(settings),
        integrations=FakeIntegrations(),
        config=Config(personality=personality),
        send_json=collector.send_json,
        send_bytes=collector.send_bytes,
        user_id="user-1",
        actions=actions,
    )
    session.actions_store = actions  # convenience for the tests
    return session, collector


class SlowTool(Tool):
    pass


def _danger_tool(calls: list[dict]) -> Tool:
    async def handler(ctx, args):
        calls.append(args)
        return ToolResult(ok=True, summary="done: " + str(args))

    return Tool(
        name="danger_thing",
        description="does a dangerous thing",
        parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
        handler=handler,
        requires_confirmation=True,
        confirmation_prompt=lambda args: f"Really do danger thing with {args.get('x')}?",
        danger=True,
    )


def _slow_tool(delay: float) -> Tool:
    async def handler(ctx, args):
        await asyncio.sleep(delay)
        return ToolResult(ok=True, summary="slow done")

    return Tool(
        name="slow_tool",
        description="slow tool",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


async def test_simple_turn_streams_sentences_and_audio(tmp_path):
    llm = FakeLLM(
        [
            [
                TextDelta("Hey there. "),
                TextDelta("How can I help?"),
                StreamDone(text="Hey there. How can I help?"),
            ]
        ]
    )
    tts = FakeTTS()
    session, collector = make_session(tmp_path, llm=llm, tts=tts)

    await session.handle_message({"type": "text", "text": "hi"})
    await session.turn_task

    assert collector.types().count("assistant.delta") == 2
    assert collector.last("assistant.message")["text"] == "Hey there. How can I help?"
    # sentence chunks may be merged into one TTS request (preferred: smoother prosody)
    spoken = " ".join(collector.audio_texts())
    assert "Hey there." in spoken and "How can I help?" in spoken
    assert len(collector.of("audio.end")) >= 1
    assert "Hey there." in " ".join(tts.texts)
    assert collector.of("tool.call") == []
    # history: system + user + assistant
    assert [m["role"] for m in session.messages] == ["system", "user", "assistant"]
    assert collector.last("error") is None


async def test_tool_call_then_final_answer(tmp_path):
    calls: list[dict] = []

    async def handler(ctx, args):
        calls.append(args)
        return ToolResult(ok=True, summary="7 emails from Sarah", display={"kind": "email_list", "items": []})

    tool = Tool(
        name="gmail_search",
        description="search mail",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        handler=handler,
    )
    llm = FakeLLM(
        [
            [StreamDone(tool_calls=[ToolCall(id="c1", name="gmail_search", arguments='{"query":"from:sarah"}')])],
            [TextDelta("You have seven from Sarah."), StreamDone(text="You have seven from Sarah.")],
        ]
    )
    session, collector = make_session(tmp_path, llm=llm, tools=[tool])

    await session.handle_message({"type": "text", "text": "any mail from sarah?"})
    await session.turn_task

    assert calls == [{"query": "from:sarah"}]
    assert collector.of("tool.call")[0]["name"] == "gmail_search"
    assert collector.of("tool.result")[0]["ok"] is True
    assert collector.last("assistant.message")["text"] == "You have seven from Sarah."
    # the tool result must be in the model context for the second call
    roles = [m["role"] for m in llm.requests[1]]
    assert roles[-1] == "tool"
    assert llm.requests[1][-1]["tool_call_id"] == "c1"
    assert collector.last("error") is None


async def test_confirmation_pauses_and_resumes_on_yes(tmp_path):
    calls: list[dict] = []
    llm = FakeLLM(
        [
            [StreamDone(tool_calls=[ToolCall(id="c1", name="danger_thing", arguments='{"x":1}')])],
            [TextDelta("All done."), StreamDone(text="All done.")],
        ]
    )
    session, collector = make_session(tmp_path, llm=llm, tools=[_danger_tool(calls)])

    await session.handle_message({"type": "text", "text": "delete everything"})
    await session.turn_task

    request = collector.last("confirm.request")
    assert request is not None and request["danger"] is True
    assert session.pending is not None
    assert session.state == "awaiting_confirmation"
    assert calls == []  # not executed yet
    assert collector.of("tool.result") == []
    assert collector.audio_texts()[-1] == "Really do danger thing with 1?"

    await session.handle_message({"type": "text", "text": "yes go ahead"})
    await session.turn_task

    assert calls == [{"x": 1}]
    assert collector.last("confirm.resolved")["approved"] is True
    assert collector.last("tool.result")["ok"] is True
    assert collector.last("assistant.message")["text"] == "All done."
    assert session.pending is None


async def test_confirmation_denied_sends_declined_result(tmp_path):
    calls: list[dict] = []
    llm = FakeLLM(
        [
            [StreamDone(tool_calls=[ToolCall(id="c1", name="danger_thing", arguments='{"x":2}')])],
            [TextDelta("Okay, I won't."), StreamDone(text="Okay, I won't.")],
        ]
    )
    session, collector = make_session(tmp_path, llm=llm, tools=[_danger_tool(calls)])

    await session.handle_message({"type": "text", "text": "do the danger thing"})
    await session.turn_task
    assert session.pending is not None
    await session.handle_message({"type": "confirm", "id": session.pending.id, "approved": False})
    await session.turn_task

    assert calls == []
    assert collector.last("confirm.resolved")["approved"] is False
    result = collector.last("tool.result")
    assert result["ok"] is False and result["error"] == "user_declined"
    # the model was told the user declined
    tool_message = [m for m in llm.requests[1] if m["role"] == "tool"][0]
    assert "user_declined" in tool_message["content"]


async def test_off_topic_answer_drops_pending_and_runs_new_turn(tmp_path):
    calls: list[dict] = []
    llm = FakeLLM(
        [
            [StreamDone(tool_calls=[ToolCall(id="c1", name="danger_thing", arguments='{"x":3}')])],
            [TextDelta("Sure, here is the weather."), StreamDone(text="Sure, here is the weather.")],
        ]
    )
    session, collector = make_session(tmp_path, llm=llm, tools=[_danger_tool(calls)])
    llm.complete_answer = "OTHER"

    await session.handle_message({"type": "text", "text": "do it"})
    await session.turn_task
    assert session.pending is not None

    await session.handle_message({"type": "text", "text": "what is the weather like in Lisbon tomorrow"})
    await session.turn_task

    assert calls == []
    assert session.pending is None
    assert collector.last("assistant.message")["text"] == "Sure, here is the weather."
    assert collector.last("confirm.resolved")["approved"] is False


async def test_barge_in_cancels_turn_and_flushes_audio(tmp_path):
    llm = FakeLLM([[TextDelta("one "), 0.2, TextDelta("two "), 0.2, StreamDone(text="one two")]])
    session, collector = make_session(tmp_path, llm=llm)
    await session.handle_message({"type": "text", "text": "talk a lot"})
    await asyncio.sleep(0.05)

    await session.handle_message({"type": "barge_in"})

    assert collector.last("cancel.audio") is not None
    assert collector.last("state")["state"] == "listening"
    assert session.turn_task is None
    assert session.token is None


async def test_slow_tool_triggers_filler_speech(tmp_path):
    llm = FakeLLM(
        [
            [StreamDone(tool_calls=[ToolCall(id="c1", name="slow_tool", arguments="{}")])],
            [TextDelta("Done now."), StreamDone(text="Done now.")],
        ]
    )
    session, collector = make_session(tmp_path, llm=llm, tools=[_slow_tool(0.4)])
    filler = session.personality.tool_filler

    await session.handle_message({"type": "text", "text": "do the slow thing"})
    await session.turn_task

    assert filler in collector.audio_texts()
    assert collector.audio_texts()[-1] == "Done now."


async def test_unknown_tool_is_reported_not_fatal(tmp_path):
    llm = FakeLLM(
        [
            [StreamDone(tool_calls=[ToolCall(id="c9", name="does_not_exist", arguments="{}")])],
            [TextDelta("Sorry, that tool is gone."), StreamDone(text="Sorry, that tool is gone.")],
        ]
    )
    session, collector = make_session(tmp_path, llm=llm)
    await session.handle_message({"type": "text", "text": "call the missing tool"})
    await session.turn_task
    result = collector.of("tool.result")[0]
    assert result["ok"] is False
    assert collector.last("assistant.message")["text"] == "Sorry, that tool is gone."


async def test_voice_utterance_goes_through_stt(tmp_path):
    llm = FakeLLM([[TextDelta("Hi."), StreamDone(text="Hi.")]])
    stt = FakeSTT("what time is it")
    session, collector = make_session(tmp_path, llm=llm, stt=stt)

    await session.handle_message({"type": "utterance.start"})
    await session.on_audio(b"\x00\x00" * 8000)  # 0.5 s of silence
    await session.handle_message({"type": "utterance.end"})
    await session.turn_task

    assert stt.calls >= 1
    assert collector.last("stt.final")["text"] == "what time is it"
    assert collector.of("user.text")[0] == {"type": "user.text", "text": "what time is it", "source": "voice"}
    assert llm.requests[0][-1] == {"role": "user", "content": "what time is it"}


async def test_short_blip_is_ignored(tmp_path):
    stt = FakeSTT("ignored")
    session, collector = make_session(tmp_path, stt=stt)
    await session.handle_message({"type": "utterance.start"})
    await session.on_audio(b"\x00\x00" * 100)
    await session.handle_message({"type": "utterance.end"})
    assert stt.calls == 0
    assert collector.of("stt.final") == []


async def test_draft_becomes_an_action_for_the_dashboard(tmp_path):
    async def draft(ctx, args):
        return ToolResult(
            ok=True,
            summary="Draft created.",
            display={
                "kind": "email_draft",
                "draftId": "draft_9",
                "to": ["sarah@acme.com"],
                "subject": "Re: lease",
                "body": "Tuesday works.",
            },
        )

    maker = Tool(name="gmail_draft", description="d", parameters={"type": "object", "properties": {}}, handler=draft)
    llm = FakeLLM([
        [StreamDone(tool_calls=[ToolCall(id="c1", name="gmail_draft", arguments="{}")])],
        [TextDelta("Draft is ready."), StreamDone(text="Draft is ready.")],
    ])
    session, collector = make_session(tmp_path, llm=llm, tools=[maker])

    await session.handle_message({"type": "text", "text": "draft a reply to sarah"})
    await session.turn_task

    stored = session.actions_store.list("user-1")
    assert len(stored) == 1
    assert stored[0].kind == "draft"
    assert stored[0].title == "Re: lease"
    # drafts land on the Drafts label — per-draft compose links are unreliable
    assert stored[0].url.endswith("/#drafts")
    created = collector.of("action.created")
    assert created and created[0]["open"] == 1
    assert created[0]["action"]["subtitle"] == "to sarah@acme.com"


async def test_failed_tools_do_not_create_actions(tmp_path):
    async def broken(ctx, args):
        return ToolResult(ok=False, error="nope", display={"kind": "email_draft", "draftId": "d1", "subject": "x"})

    maker = Tool(name="gmail_draft", description="d", parameters={"type": "object", "properties": {}}, handler=broken)
    llm = FakeLLM([
        [StreamDone(tool_calls=[ToolCall(id="c1", name="gmail_draft", arguments="{}")])],
        [TextDelta("That failed."), StreamDone(text="That failed.")],
    ])
    session, _ = make_session(tmp_path, llm=llm, tools=[maker])
    await session.handle_message({"type": "text", "text": "draft it"})
    await session.turn_task
    assert session.actions_store.count("user-1") == 0


async def test_personality_switch_rewrites_system_prompt(tmp_path):
    llm = FakeLLM([[TextDelta("Hi."), StreamDone(text="Hi.")]])
    session, collector = make_session(tmp_path, llm=llm)
    before = session.messages[0]["content"]

    session.apply_config({"personality": "sable"})
    after = session.messages[0]["content"]

    assert "You are Otto" in before
    assert "You are Sable" in after
    assert session.config.personality == "sable"


async def test_history_is_trimmed_without_orphan_tool_messages(tmp_path):
    session, _ = make_session(tmp_path)
    session.messages = [{"role": "system", "content": "s"}]
    for i in range(40):
        session.messages.append({"role": "user", "content": f"u{i}"})
        session.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
            }
        )
        session.messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "{}"})
        session.messages.append({"role": "assistant", "content": f"a{i}"})
    session.trim_history()
    assert session.messages[0]["role"] == "system"
    assert session.messages[1]["role"] != "tool"
    assert len(session.messages) <= 50


async def test_confirmation_runs_the_mcp_tool_after_yes(tmp_path):
    """Approving must run an MCP tool — the registry alone does not know it.

    Regression: the confirmation resume path resolved the tool through the local
    registry only, so every approved MCP action died as "Unknown tool".
    """
    calls: list[dict] = []

    async def handler(ctx, args):
        calls.append(dict(args))
        return ToolResult(ok=True, summary="Created 'Meet boss'.")

    mcp_tool = Tool(
        name="calendar_create_event",
        description="d",
        parameters={"type": "object", "properties": {}},
        handler=handler,
        requires_confirmation=True,
        confirmation_prompt="Create it?",
    )

    class StubMcp:
        authoritative = False

        async def discover(self):
            return []

        def get(self, name):
            return mcp_tool if name == "calendar_create_event" else None

        def prompt_note(self, send_available=False):
            return ""

        def drain_notices(self):
            return []

    llm = FakeLLM(
        [
            [StreamDone(tool_calls=[ToolCall(id="c1", name="calendar_create_event", arguments='{"summary":"Meet boss"}')])],
            [TextDelta("Done."), StreamDone(text="Done.")],
        ]
    )
    session, collector = make_session(tmp_path, llm=llm)
    session.mcp = StubMcp()

    await session.handle_message({"type": "text", "text": "add a meeting with my boss"})
    await session.turn_task
    assert session.pending is not None, "the confirmation did not pause the turn"
    assert calls == []

    await session.handle_message({"type": "text", "text": "yes"})
    await session.turn_task

    assert calls and calls[0]["summary"] == "Meet boss", "the approved MCP tool never ran"
    assert collector.last("tool.result")["ok"] is True
    assert collector.last("assistant.message")["text"] == "Done."


def test_reasoning_effort_flows_to_the_llm(tmp_path):
    """The Thinking selector: "" leaves the per-model default, a value overrides it."""
    llm = FakeLLM([[TextDelta("hi"), StreamDone(text="hi")]])
    session, _ = make_session(tmp_path, llm=llm)
    assert session._llm_extra() == {}

    session.apply_config({"reasoning_effort": "medium"})
    assert session._llm_extra() == {"reasoning_effort": "medium"}

    session.apply_config({"reasoning_effort": "  "})
    assert session._llm_extra() == {}

    assert session.config.to_dict()["reasoning_effort"] == ""


def test_confirmation_ids_speak_as_titles(tmp_path):
    """A confirmation question never reads a raw id when memory knows the title."""
    session, _ = make_session(tmp_path)
    session.memory.add("event", "ku5mdqp4bmnjc66ppqvl8juv6o", "Otto reminder test")

    human = session._humanize_ids("I'll update ku5mdqp4bmnjc66ppqvl8juv6o. Okay?")
    assert human == "I'll update 'Otto reminder test'. Okay?"

    preview = session._enrich_preview(
        {"kind": "event_create", "title": "", "start": ""},
        {"eventId": "ku5mdqp4bmnjc66ppqvl8juv6o"},
    )
    assert preview["title"] == "Otto reminder test"


def test_system_prompt_carries_the_users_name(tmp_path):
    """Once people_profile has cached the name, every session signs drafts with it."""
    session, _ = make_session(tmp_path)

    class Users:
        saved: dict = {}

        def set_secret(self, user_id, key, value):
            self.saved[(user_id, key)] = value

        def get_secret(self, user_id, key):
            return self.saved.get((user_id, key))

    session.integrations.users = Users()
    session.integrations.user_id = "u1"
    session.integrations.users.set_secret("u1", "profile_name", "Guy Sandler")
    session._build_system_message(replace=True)
    system = session.messages[0]["content"]
    assert "The user's name is Guy Sandler" in system
    assert "sign drafted email" in system
