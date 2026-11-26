"""Parallel subagents.

A subagent is the same model in a *private* conversation with a restricted tool
belt: no audio, no confirmation prompts, a small step budget. It exists so the
main conversation can split a job ("compare these three invoices", "draft a
reply and a summary") across several of those private loops at once and only
read the conclusions.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .pipeline.llm import LLMError, ReasoningDelta, StreamDone, TextDelta
from .tools.base import ToolContext, ToolResult

log = logging.getLogger("otto.agents")

#: read-only tools a subagent may use; anything that writes or needs a human is excluded
DEFAULT_ALLOWLIST = (
    "gmail_search",
    "gmail_read",
    "calendar_list_events",
    "slack_search",
    "slack_read_channel",
    "slack_list_channels",
    "get_current_time",
    "recall",
)

SUBAGENT_SYSTEM = """You are a focused research assistant working for a voice assistant.
You have a small set of read-only tools. Do the task, then answer in plain text.

Rules:
- Be exhaustive but compact: the answer is read by another model, not a human.
- Include the concrete facts asked for (names, dates, ids, numbers, quotes).
- Never invent data. If a tool returns nothing, say so.
- No markdown headings, no emoji. Short lines are fine.
"""


@dataclass(slots=True)
class SubagentResult:
    title: str
    ok: bool
    text: str
    steps: int = 0
    tool_calls: list[str] = field(default_factory=list)
    ms: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "ok": self.ok,
            "result": self.text,
            "steps": self.steps,
            "tools": self.tool_calls,
            "ms": self.ms,
            "error": self.error,
        }


class SubagentRunner:
    """Runs one private tool loop; call many at once for parallel work."""

    def __init__(self, llm, settings, registry, *, integrations: Any = None, mcp: Any = None) -> None:
        self.llm = llm
        self.settings = settings
        self.registry = registry
        self.integrations = integrations
        self.mcp = mcp
        cfg = settings.cfg("agents", default={}) or {}
        self.max_steps = int(cfg.get("max_steps", 4))
        self.timeout_s = float(cfg.get("timeout_s", 120))
        allow = cfg.get("tool_allowlist") or DEFAULT_ALLOWLIST
        self.allowlist = [str(name) for name in allow]

    # ------------------------------------------------------------------ tools
    def tools(self, ctx: ToolContext | None = None) -> list[dict[str, Any]]:
        out = []
        for name in self.allowlist:
            tool = self.registry.get(name)
            if tool is None and self.mcp is not None:
                tool = self.mcp.get(name)
            if tool is None:
                continue
            try:
                if not tool.is_available(ctx):
                    continue
            except Exception:  # noqa: BLE001
                continue
            if tool.needs_confirmation({}):
                continue
            out.append(tool.to_schema())
        return out

    # ------------------------------------------------------------------- run
    async def run(
        self,
        title: str,
        instructions: str,
        *,
        model: str,
        context: str = "",
        max_chars: int = 1200,
    ) -> SubagentResult:
        started = time.perf_counter()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SUBAGENT_SYSTEM + (f"\n\nShared context:\n{context}" if context else "")},
            {"role": "user", "content": instructions},
        ]
        tool_ctx = ToolContext(settings=self.settings, integrations=self.integrations, session=None, extras={"subagent": True})
        schemas = self.tools(tool_ctx)
        calls: list[str] = []
        steps = 0
        try:
            final_text = ""
            for _ in range(self.max_steps):
                steps += 1
                text_parts: list[str] = []
                tool_calls = []
                async for event in self.llm.stream(
                    messages,
                    model=model,
                    tools=schemas or None,
                    temperature=0.4,
                    max_tokens=700,
                ):
                    if isinstance(event, TextDelta):
                        text_parts.append(event.text)
                    elif isinstance(event, StreamDone):
                        tool_calls = event.tool_calls
                text = "".join(text_parts).strip()
                if not tool_calls:
                    final_text = text
                    break
                messages.append(
                    {
                        "role": "assistant",
                        "content": text or None,
                        "tool_calls": [
                            {"id": call.id, "type": "function",
                             "function": {"name": call.name, "arguments": call.arguments}}
                            for call in tool_calls
                        ],
                    }
                )
                for call in tool_calls:
                    tool = self.registry.get(call.name) or (self.mcp.get(call.name) if self.mcp is not None else None)
                    calls.append(call.name)
                    if tool is None or call.name not in self.allowlist:
                        result = ToolResult(ok=False, error=f"tool '{call.name}' is not available to subagents")
                    else:
                        result = await tool.invoke(tool_ctx, call.args())
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": result.to_model_payload(limit=4000),
                        }
                    )
            else:
                final_text = text or "(step budget exhausted)"

            return SubagentResult(
                title=title,
                ok=True,
                text=final_text[:max_chars],
                steps=steps,
                tool_calls=calls,
                ms=int((time.perf_counter() - started) * 1000),
            )
        except (LLMError, asyncio.TimeoutError) as exc:
            return SubagentResult(
                title=title,
                ok=False,
                text="",
                steps=steps,
                tool_calls=calls,
                ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("subagent failed")
            return SubagentResult(
                title=title,
                ok=False,
                text="",
                steps=steps,
                tool_calls=calls,
                ms=int((time.perf_counter() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )

    async def run_many(self, tasks: list[tuple[str, str]], *, model: str, context: str = "") -> list[SubagentResult]:
        cfg = self.settings.cfg("agents", default={}) or {}
        limit = max(1, int(cfg.get("max_parallel", 4)))
        semaphore = asyncio.Semaphore(limit)

        async def one(title: str, instructions: str) -> SubagentResult:
            async with semaphore:
                return await asyncio.wait_for(
                    self.run(title, instructions, model=model, context=context),
                    timeout=self.timeout_s,
                )

        results = await asyncio.gather(
            *(one(title, instructions) for title, instructions in tasks),
            return_exceptions=True,
        )
        out: list[SubagentResult] = []
        for (title, _), result in zip(tasks, results):
            if isinstance(result, BaseException):
                out.append(SubagentResult(title=title, ok=False, text="", error=str(result)))
            else:
                out.append(result)
        return out
