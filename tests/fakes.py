"""Test doubles for the pipeline (LLM, STT, TTS) and a message collector."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from otto.pipeline.llm import StreamDone, TextDelta
from otto.pipeline.stt import Transcript
from otto.pipeline.tts import TTSChunk
from otto.protocol import ToolCall, parse_audio_frame


class FakeLLM:
    """Scripted LLM. Each script is a list of pipeline events; a float inserts a sleep."""

    def __init__(self, scripts: list[list[Any]] | None = None, *, complete_answer: str = "APPROVE") -> None:
        self.scripts = list(scripts or [])
        self.complete_answer = complete_answer
        self.requests: list[list[dict[str, Any]]] = []
        self.complete_calls = 0
        self.closed = 0

    @property
    def configured(self) -> bool:
        return True

    async def stream(self, messages, *, model, tools=None, temperature=0.7, max_tokens=500, extra_body=None):
        self.requests.append([dict(m) for m in messages])
        script = self.scripts.pop(0) if self.scripts else [StreamDone(text="")]
        try:
            for item in script:
                if isinstance(item, (int, float)):
                    await asyncio.sleep(float(item))
                    continue
                yield item
        finally:
            self.closed += 1

    async def complete(self, messages, *, model, max_tokens=16, temperature=0.0):
        self.complete_calls += 1
        return self.complete_answer

    async def models(self, *, refresh: bool = False):
        return [{"id": "fake-model", "label": "fake-model", "reasoning": False}]

    def background_variant(self, model: str) -> str:
        return f"{model}-background" if getattr(self, "has_background", False) else model

    def has_background_variant(self, model: str) -> bool:
        return self.background_variant(model) != model


class FakeSTT:
    def __init__(self, text: str = "hello there") -> None:
        self.text = text
        self.calls = 0

    @property
    def configured(self) -> bool:
        return True

    async def transcribe(self, pcm: bytes, *, sample_rate: int = 16000, language=None, model=None, timeout: float = 30.0):
        self.calls += 1
        return Transcript(text=self.text, latency_ms=12, duration_ms=int(len(pcm) / 32))


class FakeTTS:
    def __init__(self, chunk_samples: int = 2400, sample_rate: int = 24000) -> None:
        self.texts: list[str] = []
        self.chunk_samples = chunk_samples
        self.sample_rate = sample_rate
        self.aborted = 0

    @property
    def enabled(self) -> bool:
        return True

    async def stream(self, text, *, voice=None, instruct=None, language=None, speed=1.0) -> AsyncIterator[TTSChunk]:
        self.texts.append(text)
        try:
            for _ in range(2):
                yield TTSChunk(pcm=b"\x01\x00" * self.chunk_samples, sample_rate=self.sample_rate)
                await asyncio.sleep(0)
        finally:
            self.aborted += 1

    async def health(self, *, refresh: bool = False):
        return {"ok": True, "model": "fake"}

    async def voices(self, *, refresh: bool = False):
        return [{"id": "Aiden", "description": "fake"}]


class Collector:
    """Captures everything the session sends to the browser."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.frames: list[bytes] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append(data)

    # ---------------------------------------------------------------- helpers
    def types(self) -> list[str]:
        return [event["type"] for event in self.events]

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["type"] == kind]

    def last(self, kind: str) -> dict[str, Any] | None:
        for event in reversed(self.events):
            if event["type"] == kind:
                return event
        return None

    def audio_segments(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for frame in self.frames:
            parsed = parse_audio_frame(frame)
            assert parsed is not None
            seq, _rate, _index, _pcm = parsed
            out.append((seq, len(frame)))
        return out

    def audio_texts(self) -> list[str]:
        return [event["text"] for event in self.events if event["type"] == "audio.start"]
