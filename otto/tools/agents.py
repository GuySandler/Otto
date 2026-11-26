"""Parallel subagents and background jobs as tools.

Two different shapes of "get more done without holding the conversation":

* ``delegate_task`` — split one request into N independent sub-questions, run
  them in parallel private loops, and hand the main model the conclusions.
  Gated behind the *parallel agents* toggle.
* ``start_background_task`` — kick off longer work and keep talking; the result
  is announced out loud when it lands.
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..jobs import FAILED, RUNNING
from .base import Tool, ToolContext, ToolResult

MAX_TASKS = 6


def _session(ctx: ToolContext) -> Any:
    session = ctx.session
    if session is None or not hasattr(session, "subagents"):
        raise RuntimeError("this tool only works inside a live conversation")
    return session


async def _delegate(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    session = _session(ctx)
    raw = args.get("tasks") or []
    tasks: list[tuple[str, str]] = []
    if isinstance(raw, list):
        for index, item in enumerate(raw[:MAX_TASKS]):
            if isinstance(item, str):
                tasks.append((f"part {index + 1}", item))
            elif isinstance(item, dict) and item.get("instructions"):
                tasks.append((str(item.get("title") or f"part {index + 1}")[:60], str(item["instructions"])))
    if not tasks:
        return ToolResult(ok=False, error="no tasks given")

    started = asyncio.get_running_loop().time()
    results = await session.subagents.run_many(
        tasks,
        model=session.agent_model(),
        context=session.memory.prompt_block(max_lines=8),
    )
    elapsed = asyncio.get_running_loop().time() - started

    blocks: list[str] = []
    display_items: list[dict[str, Any]] = []
    for result in results:
        status = "ok" if result.ok else f"failed: {result.error}"
        body = result.text if result.ok else ""
        blocks.append(f"[{result.title}] ({status}, {result.ms} ms)\n{body}")
        display_items.append(result.as_dict())

    ok = any(result.ok for result in results)
    summary = (
        f"{len(results)} subagent(s) finished in {elapsed:.1f}s.\n\n" + "\n\n".join(blocks)
    )
    return ToolResult(
        ok=ok,
        summary=summary[:6000],
        display={
            "kind": "subagents",
            "items": display_items,
            "elapsedMs": int(elapsed * 1000),
        },
    )


async def _background(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    session = _session(ctx)
    title = str(args.get("title") or "background task").strip()[:80]
    instructions = str(args.get("instructions") or "").strip()
    if not instructions:
        return ToolResult(ok=False, error="instructions are required")

    model = session.agent_model()
    context = session.memory.prompt_block(max_lines=8)

    async def factory() -> str:
        result = await session.subagents.run(title, instructions, model=model, context=context)
        if not result.ok:
            raise RuntimeError(result.error or "subagent failed")
        return result.text

    job = session.jobs.start(title, factory, kind="subagent", meta={"instructions": instructions})
    return ToolResult(
        ok=True,
        summary=(
            f"Started background task {job.id} ('{title}'). It runs detached; you will be told when "
            "it finishes. Do not wait for it — keep talking to the user."
        ),
        display={"kind": "job_started", "id": job.id, "title": title, "status": job.status},
        data={"job_id": job.id},
    )


async def _list_tasks(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    session = _session(ctx)
    jobs = session.jobs.list()
    if not jobs:
        return ToolResult(ok=True, summary="No background tasks have been started.", display={"kind": "jobs", "items": []})
    lines = []
    for job in jobs[:10]:
        if job.status in (RUNNING, "queued"):
            lines.append(f"- {job.id} '{job.title}': {job.status}")
        elif job.status == FAILED:
            lines.append(f"- {job.id} '{job.title}': failed ({job.error})")
        else:
            lines.append(f"- {job.id} '{job.title}': done — {job.result[:400]}")
    return ToolResult(
        ok=True,
        summary="\n".join(lines),
        display={"kind": "jobs", "items": [job.to_event() for job in jobs[:10]]},
    )


TOOLS: list[Tool] = [
    Tool(
        name="delegate_task",
        description=(
            "Split a request into independent parts and run them in parallel subagents, then read "
            "their findings. Use it when the user explicitly wants several things looked at at once "
            "('check these three threads and tell me what matters', 'compare the two quotes'). "
            "Give each task a short title and self-contained instructions — subagents do not see "
            "this conversation. Max 6 tasks."
        ),
        parameters={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "Independent sub-tasks to run in parallel.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Short label, e.g. 'invoice 1'."},
                            "instructions": {
                                "type": "string",
                                "description": "Self-contained instructions for that subagent.",
                            },
                        },
                        "required": ["title", "instructions"],
                    },
                }
            },
            "required": ["tasks"],
        },
        handler=_delegate,
        category="agents",
        flag="parallel_agents",
        timeout_s=180.0,
    ),
    Tool(
        name="start_background_task",
        description=(
            "Start a long-running job in the background and return immediately. Use it for work "
            "that would otherwise stall the conversation (reading many messages, building a plan, "
            "preparing a summary). The result is announced to the user automatically when it lands, "
            "so never wait for it and never invent its result."
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short human title shown in the UI."},
                "instructions": {"type": "string", "description": "Self-contained instructions for the job."},
            },
            "required": ["title", "instructions"],
        },
        handler=_background,
        category="agents",
        timeout_s=20.0,
    ),
    Tool(
        name="list_tasks",
        description="List background tasks and the results of the ones that already finished.",
        parameters={"type": "object", "properties": {}},
        handler=_list_tasks,
        category="agents",
        timeout_s=10.0,
    ),
]
