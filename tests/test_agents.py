"""Parallel subagents, background jobs, toggles and reference resolution."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
import yaml

from otto.jobs import DONE, JobManager
from otto.memory import KIND_EVENT
from otto.personality import PersonalityStore
from otto.pipeline.llm import StreamDone, TextDelta
from otto.protocol import Config, ToolCall
from otto.session import Session
from otto.settings import Settings
from otto.tools.base import Tool, ToolContext, ToolResult
from otto.tools.registry import ToolRegistry

from .fakes import Collector, FakeLLM, FakeSTT, FakeTTS

REPO = Path(__file__).resolve().parent.parent


class RouterLLM:
    """Answers each subagent based on a keyword in its instructions."""

    def __init__(self, routes: dict[str, str], *, main_script: list | None = None) -> None:
        self.routes = routes
        self.calls: list[list[dict]] = []
        self.main_script = list(main_script or [])
        self.has_background = True

    @property
    def configured(self) -> bool:
        return True

    def background_variant(self, model: str) -> str:
        return f"{model}-background"

    def _reply(self, messages: list[dict]) -> str:
        joined = " ".join(str(m.get("content") or "") for m in messages)
        for needle, answer in self.routes.items():
            if needle in joined:
                return answer
        return "nothing found"

    async def stream(self, messages, *, model, tools=None, temperature=0.7, max_tokens=500, extra_body=None):
        self.calls.append([dict(m) for m in messages])
        if self.calls[-1] and self.calls[-1][0].get("role") == "system" and "focused research assistant" in str(self.calls[-1][0].get("content")):
            text = self._reply(messages)
            yield TextDelta(text)
            yield StreamDone(text=text)
            return
        script = self.main_script.pop(0) if self.main_script else [StreamDone(text="")]
        for item in script:
            yield item

    async def complete(self, messages, *, model, max_tokens=16, temperature=0.0):
        return "APPROVE"

    async def models(self, *, refresh: bool = False):
        return [{"id": "fake-model", "label": "fake-model"}]


class FakeGoogle:
    def __init__(self) -> None:
        self.patched: list[dict] = []

    async def calendar_patch(self, event_id, **fields):
        self.patched.append({"event_id": event_id, **fields})
        return {"id": event_id, "summary": fields.get("summary") or "Design sync",
                "start": fields.get("start") or "", "end": fields.get("end") or ""}

    async def calendar_delete(self, event_id):
        return {"id": event_id, "deleted": True}


class FakeIntegrations:
    def __init__(self) -> None:
        self.google = FakeGoogle()
        self.slack = None

    def status(self):
        return {}

    async def aclose(self):
        return None


def make_session(tmp_path: Path, *, llm, tools: list[Tool] | None = None, integrations=None, config: Config | None = None):
    shutil.copytree(REPO / "config", tmp_path / "config", dirs_exist_ok=True)
    app_cfg = yaml.safe_load((REPO / "config" / "app.yaml").read_text())
    # unit tests are offline: MCP is exercised separately with a fake server
    app_cfg["mcp"] = {"enabled": False}
    settings = Settings(root=tmp_path, lunaroute_api_key="k", app_cfg=app_cfg)
    registry = ToolRegistry.discover()
    for tool in tools or []:
        registry.add(tool)
    collector = Collector()
    session = Session(
        session_id="t",
        settings=settings,
        llm=llm,
        stt=FakeSTT(),
        tts=FakeTTS(),
        registry=registry,
        personalities=PersonalityStore(settings),
        integrations=integrations or FakeIntegrations(),
        config=config or Config(),
        send_json=collector.send_json,
        send_bytes=collector.send_bytes,
    )
    return session, collector


async def test_delegate_task_is_gated_by_the_toggle(tmp_path):
    llm = RouterLLM({"A": "a"}, main_script=[[StreamDone(text="ok")]])
    session, _ = make_session(tmp_path, llm=llm)

    assert "delegate_task" not in [s["function"]["name"] for s in session.registry.schemas(flags=session.flags())]
    session.config.parallel_agents = True
    assert "delegate_task" in [s["function"]["name"] for s in session.registry.schemas(flags=session.flags())]
    # background work is always available
    names = [s["function"]["name"] for s in session.registry.schemas(flags=session.flags())]
    assert "start_background_task" in names and "list_tasks" in names


async def test_delegate_task_runs_subagents_in_parallel(tmp_path):
    llm = RouterLLM({"ALPHA": "alpha result", "BETA": "beta result"})
    session, _ = make_session(tmp_path, llm=llm, config=Config(parallel_agents=True))
    tool = session.registry.get("delegate_task")
    ctx = session.tool_context

    started = asyncio.get_running_loop().time()
    result = await tool.invoke(
        ctx,
        {"tasks": [
            {"title": "alpha", "instructions": "look at ALPHA"},
            {"title": "beta", "instructions": "look at BETA"},
        ]},
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert result.ok
    assert "alpha result" in result.summary and "beta result" in result.summary
    assert result.display["kind"] == "subagents"
    assert [item["title"] for item in result.display["items"]] == ["alpha", "beta"]
    assert elapsed < 3.0  # both ran concurrently, not sequentially


async def test_subagent_tool_allowlist_blocks_writes(tmp_path):
    llm = RouterLLM({"X": "x"})
    session, _ = make_session(tmp_path, llm=llm)
    names = [schema["function"]["name"] for schema in session.subagents.tools()]
    assert "gmail_search" not in names  # not configured in tests -> unavailable
    assert "delegate_task" not in names and "start_background_task" not in names
    assert "stop" not in names


async def test_background_job_is_announced_out_loud(tmp_path):
    llm = RouterLLM(
        {"SUMMARISE": "three invoices, two overdue"},
        main_script=[
            # announcement turn produced by the main model
            [TextDelta("The invoice check finished: two are overdue."), StreamDone(text="The invoice check finished: two are overdue.")],
        ],
    )
    session, collector = make_session(tmp_path, llm=llm)
    tool = session.registry.get("start_background_task")

    result = await tool.invoke(session.tool_context, {"title": "invoice check", "instructions": "SUMMARISE the invoices"})
    assert result.ok and result.display["kind"] == "job_started"
    job_id = result.data["job_id"]

    for _ in range(120):
        await asyncio.sleep(0.05)
        job = session.jobs.get(job_id)
        if job and job.status == DONE and collector.last("assistant.message"):
            break

    assert session.jobs.get(job_id).status == DONE
    assert "two overdue" in session.jobs.get(job_id).result
    updates = [e for e in collector.of("job.update")]
    assert {u["status"] for u in updates} >= {"running", "done"}
    # the result was spoken as its own turn
    assert collector.last("assistant.message")["text"].startswith("The invoice check finished")
    # and the internal note is in the history (system-facing) but was not shown as user text
    assert any("background task finished" in str(m.get("content")) for m in session.messages)
    assert [e for e in collector.of("user.text")] == []


async def test_job_manager_cancellation():
    settings = Settings(app_cfg={"jobs": {"max_concurrent": 2}})
    events: list[str] = []

    async def slow():
        await asyncio.sleep(5)
        return "never"

    manager = JobManager(settings, on_event=lambda job: events.append(job.status))
    job = manager.start("slow one", slow)
    await asyncio.sleep(0.05)
    assert manager.cancel(job.id) is True
    await asyncio.sleep(0.05)
    assert job.status == "cancelled"
    assert "running" in events


async def test_background_model_toggle_routes_subagents(tmp_path):
    llm = RouterLLM({})
    session, _ = make_session(tmp_path, llm=llm, config=Config(model="fake-model"))

    assert session.agent_model() == "fake-model"
    session.apply_config({"background_model": True})
    assert session.agent_model() == "fake-model-background"


async def test_memory_block_reaches_the_model_on_the_next_turn(tmp_path):
    """The ids from turn 1 must be visible to the model in turn 2."""
    async def fake_list(ctx, args):
        return ToolResult(
            ok=True,
            summary="1 event",
            display={"kind": "event_list", "items": [
                {"id": "evt_42", "title": "Design sync", "start": "2026-11-25T10:00:00-08:00", "end": ""},
            ]},
        )

    lister = Tool(name="list_my_events", description="d", parameters={"type": "object", "properties": {}}, handler=fake_list)
    recorder: list[dict] = []

    async def fake_update(ctx, args):
        recorder.append(args)
        return ToolResult(ok=True, summary="moved")

    updater = Tool(
        name="move_event_demo",
        description="d",
        parameters={"type": "object", "properties": {"event_id": {"type": "string"}, "start": {"type": "string"}}},
        handler=fake_update,
        requires_confirmation=True,
    )

    llm = FakeLLM([
        # turn 1: list the calendar
        [StreamDone(tool_calls=[ToolCall(id="c1", name="list_my_events", arguments="{}")])],
        [TextDelta("You have a design sync tomorrow."), StreamDone(text="You have a design sync tomorrow.")],
        # turn 2: the user says "move that to four" -> the model reuses the id from memory
        [StreamDone(tool_calls=[ToolCall(
            id="c2",
            name="move_event_demo",
            arguments='{"event_id": "evt_42", "start": "2026-11-25T16:00:00-08:00"}',
        )])],
        [TextDelta("Moved."), StreamDone(text="Moved.")],
    ])
    session, collector = make_session(tmp_path, llm=llm, tools=[lister, updater])

    await session.handle_message({"type": "text", "text": "what's on my calendar tomorrow?"})
    await session.turn_task
    assert session.memory.resolve(KIND_EVENT).id == "evt_42"

    await session.handle_message({"type": "text", "text": "move that to four"})
    await session.turn_task

    system_prompt = llm.requests[2][0]["content"]
    assert "evt_42" in system_prompt and "Design sync" in system_prompt
    # the tool needed confirmation; approving runs it with the remembered id
    assert session.pending is not None
    await session.handle_message({"type": "confirm", "id": session.pending.id, "approved": True})
    await session.turn_task
    assert recorder == [{"event_id": "evt_42", "start": "2026-11-25T16:00:00-08:00"}]


async def test_calendar_update_tool_resolves_a_hint(tmp_path):
    llm = RouterLLM({})
    integrations = FakeIntegrations()
    session, _ = make_session(tmp_path, llm=llm, integrations=integrations)
    session.memory.remember_tool_display(
        {"kind": "event_list", "items": [
            {"id": "evt_1", "title": "Design sync", "start": "2026-11-25T10:00:00-08:00", "end": ""},
            {"id": "evt_2", "title": "Dentist", "start": "2026-11-26T15:00:00-08:00", "end": ""},
        ]}
    )
    tool = session.registry.get("calendar_update_event")
    assert tool is not None

    result = await tool.invoke(
        session.tool_context,
        {"hint": "the dentist appointment", "start": "2026-11-26T17:00:00-08:00"},
    )
    assert result.ok, result.error
    assert integrations.google.patched[0]["event_id"] == "evt_2"
    assert "updated" in result.summary.lower()


async def test_calendar_update_tool_without_any_reference_fails_politely(tmp_path):
    session, _ = make_session(tmp_path, llm=RouterLLM({}))
    tool = session.registry.get("calendar_update_event")
    result = await tool.invoke(session.tool_context, {"start": "2026-11-26T17:00:00-08:00"})
    assert result.ok is False
    assert "which event" in (result.error or "").lower()
