"""Lunaroute chat-completions client (streaming, tool calling)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from ..protocol import ToolCall, ToolCalls

log = logging.getLogger("otto.llm")


class LLMError(RuntimeError):
    pass


@dataclass(slots=True)
class TextDelta:
    text: str


@dataclass(slots=True)
class ReasoningDelta:
    text: str


@dataclass(slots=True)
class StreamDone:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


LLMEvent = TextDelta | ReasoningDelta | StreamDone


class LunarouteLLM:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.timeout = float(settings.cfg("lunaroute", "request_timeout_s", default=120))
        self.extra_body: dict[str, Any] = dict(settings.cfg("lunaroute", "extra_body", default={}) or {})
        self.model_options: dict[str, dict[str, Any]] = dict(
            settings.cfg("lunaroute", "model_options", default={}) or {}
        )
        self._models_cache: list[dict[str, Any]] = []
        self._model_ids: set[str] = set()

    # ------------------------------------------------------------------ client
    def _client(self) -> httpx.AsyncClient:
        headers = {"Content-Type": "application/json"}
        if self.settings.lunaroute_api_key:
            headers["Authorization"] = f"Bearer {self.settings.lunaroute_api_key}"
        return httpx.AsyncClient(
            base_url=self.settings.lunaroute_base_url,
            headers=headers,
            timeout=httpx.Timeout(self.timeout, connect=15.0),
        )

    @property
    def configured(self) -> bool:
        return bool(self.settings.lunaroute_api_key)

    # ------------------------------------------------------------------ models
    async def models(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        if self._models_cache and not refresh:
            return self._models_cache
        async with self._client() as client:
            resp = await client.get("/models")
            resp.raise_for_status()
            data = resp.json().get("data", [])
        # keep the ones that can hold a spoken conversation
        out: list[dict[str, Any]] = []
        for model in data:
            caps = model.get("capabilities") or {}
            if caps.get("transcription"):
                continue
            # classifiers / embedding-ish models are not conversational
            if model.get("type") == "classifier" or caps.get("systemone"):
                continue
            if not (caps.get("openai_chat") or caps.get("openai_responses")):
                continue
            out.append(
                {
                    "id": model["id"],
                    "label": model.get("name") or model["id"],
                    "context": model.get("context_window") or model.get("context_length"),
                    "reasoning": bool(caps.get("reasoning")),
                }
            )
        self._models_cache = out
        self._model_ids = {model["id"] for model in out}
        return out

    def background_variant(self, model: str) -> str:
        """The ``*-background`` sibling Lunaroute exposes for this model, if any."""
        suffix = str(self.settings.cfg("lunaroute", "background_suffix", default="-background") or "-background")
        if model.endswith(suffix):
            return model
        candidate = f"{model}{suffix}"
        if candidate in self._model_ids:
            return candidate
        return model

    def has_background_variant(self, model: str) -> bool:
        return self.background_variant(model) != model

    async def transcription_models(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        """Model ids that support /audio/transcriptions."""
        await self.models(refresh=refresh)
        async with self._client() as client:
            resp = await client.get("/models")
            resp.raise_for_status()
            data = resp.json().get("data", [])
        return [
            {"id": model["id"], "label": model.get("name") or model["id"]}
            for model in data
            if (model.get("capabilities") or {}).get("transcription")
        ]

    # ------------------------------------------------------------------ stream
    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        options = self.model_options.get(model) or {}
        body.update(self.extra_body)
        body.update(options.get("extra_body") or {})
        if options.get("temperature") is not None:
            body["temperature"] = float(options["temperature"])
        if options.get("max_tokens") is not None:
            body["max_tokens"] = int(options["max_tokens"])
        if extra_body:
            body.update(extra_body)

        collected = ToolCalls()
        text_parts: list[str] = []
        finish_reason = ""
        usage: dict[str, Any] = {}

        async with self._client() as client:
            try:
                async with client.stream("POST", "/chat/completions", json=body) as resp:
                    if resp.status_code >= 400:
                        detail = (await resp.aread()).decode(errors="replace")[:400]
                        raise LLMError(f"lunaroute {resp.status_code}: {detail}")
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload or payload == "[DONE]":
                            if payload == "[DONE]":
                                break
                            continue
                        try:
                            chunk = json.loads(payload)
                        except ValueError:
                            continue
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                        if reasoning:
                            yield ReasoningDelta(str(reasoning))
                        content = delta.get("content")
                        if content:
                            text_parts.append(content)
                            yield TextDelta(content)
                        for call_delta in delta.get("tool_calls") or []:
                            collected.add(call_delta)
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
            except httpx.HTTPError as exc:
                raise LLMError(f"lunaroute request failed: {exc}") from exc

        yield StreamDone(
            text="".join(text_parts),
            tool_calls=collected.finish(),
            finish_reason=finish_reason,
            usage=usage,
        )

    # ---------------------------------------------------------------- one-shot
    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        max_tokens: int = 16,
        temperature: float = 0.0,
    ) -> str:
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        async with self._client() as client:
            try:
                resp = await client.post("/chat/completions", json=body)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise LLMError(f"lunaroute request failed: {exc}") from exc
            data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content") or ""
