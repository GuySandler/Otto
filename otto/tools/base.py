"""Tool primitives.

A tool is a plain description + an async handler. Modules under ``otto/tools``
that expose a module level ``TOOLS: list[Tool]`` are discovered automatically —
that is the whole extension mechanism. Drop a file in, restart, done.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger("otto.tools")

Handler = Callable[["ToolContext", dict[str, Any]], Awaitable["ToolResult"]]


@dataclass(slots=True)
class ToolResult:
    ok: bool = True
    #: compact text handed back to the model
    summary: str = ""
    #: structured payload for the UI card
    display: dict[str, Any] | None = None
    #: raw data (kept out of the model context unless small)
    data: Any = None
    #: optional line the assistant should say verbatim (already spoken by the channel)
    speaks: str | None = None
    error: str | None = None

    def to_model_payload(self, *, limit: int = 6000) -> str:
        import json

        payload: dict[str, Any] = {"ok": self.ok}
        if self.error:
            payload["error"] = self.error
        payload["summary"] = self.summary[:limit]
        if self.data is not None:
            try:
                encoded = json.dumps(self.data, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                encoded = str(self.data)
            if len(encoded) <= limit:
                payload["data"] = self.data
        return json.dumps(payload, ensure_ascii=False, default=str)[: limit * 2]


@dataclass(slots=True)
class ToolContext:
    settings: Any
    integrations: Any = None
    session: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    def log(self, message: str, **fields: Any) -> None:
        log.info("%s %s", message, fields or "")


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    category: str = "general"
    #: True, False, or a predicate on the call arguments
    requires_confirmation: bool | Callable[[dict[str, Any]], bool] = False
    #: spoken question when confirmation is required
    confirmation_prompt: Callable[[dict[str, Any]], str] | str | None = None
    #: structured preview for the confirmation card
    preview: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    #: irreversible / money / etc — UI paints it red
    danger: bool = False
    timeout_s: float = 45.0
    #: session-config flag that must be on for the model to see this tool
    #: (e.g. "parallel_agents"); None means always available
    flag: str | None = None
    #: predicate telling whether the integration is configured (process-wide)
    available: Callable[[], bool] | None = None
    #: predicate taking the ToolContext — this is how per-user credentials are
    #: checked ("this signed-in account connected Google")
    available_for: Callable[["ToolContext"], bool] | None = None
    unavailable_hint: str = ""

    # ------------------------------------------------------------------ schema
    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_catalog(self, ctx: ToolContext | None = None) -> dict[str, Any]:
        configured = self.is_available(ctx)
        return {
            "name": self.name,
            "flag": self.flag,
            "description": self.description[:400],
            "category": self.category,
            "requires_confirmation": bool(self._confirmation_flag()),
            "danger": self.danger,
            "configured": configured,
            "unavailable_hint": self.unavailable_hint if not configured else "",
            "parameters": self.parameters,
        }

    def _confirmation_flag(self) -> bool:
        return self.requires_confirmation is True

    def is_available(self, ctx: "ToolContext | None" = None) -> bool:
        if self.available is not None:
            try:
                if not bool(self.available()):
                    return False
            except Exception:  # noqa: BLE001
                return False
        if self.available_for is not None:
            if ctx is None:
                return False
            try:
                return bool(self.available_for(ctx))
            except Exception:  # noqa: BLE001
                return False
        return True

    # ------------------------------------------------------------ confirmation
    def needs_confirmation(self, args: dict[str, Any]) -> bool:
        if callable(self.requires_confirmation):
            try:
                return bool(self.requires_confirmation(args))
            except Exception:  # noqa: BLE001
                return True
        return bool(self.requires_confirmation)

    def confirmation_text(self, args: dict[str, Any]) -> str:
        prompt = self.confirmation_prompt
        if callable(prompt):
            try:
                return str(prompt(args))
            except Exception:  # noqa: BLE001
                pass
        elif isinstance(prompt, str):
            return prompt
        summary = args.get("summary") or args.get("text") or args.get("query") or args.get("subject")
        return f"Should I run {self.name.replace('_', ' ')}" + (f" with {summary}?" if summary else "?")

    def confirmation_preview(self, args: dict[str, Any]) -> dict[str, Any] | None:
        if callable(self.preview):
            try:
                return self.preview(args)
            except Exception:  # noqa: BLE001
                return None
        return None

    # ------------------------------------------------------------------- invoke
    async def invoke(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        try:
            result = self.handler(ctx, args)
            if inspect.isawaitable(result):
                result = await asyncio.wait_for(result, timeout=self.timeout_s)
            if not isinstance(result, ToolResult):
                return ToolResult(ok=True, summary=str(result))
            return result
        except asyncio.TimeoutError:
            return ToolResult(ok=False, error=f"{self.name} timed out after {self.timeout_s:.0f}s", summary="The tool timed out.")
        except Exception as exc:  # noqa: BLE001 - tools must never crash a turn
            log.exception("tool %s failed", self.name)
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}", summary=f"The tool failed: {exc}")
