"""Wire protocol helpers shared by the server (and mirrored by web/app.js).

See docs/PROTOCOL.md for the full description.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any

AUDIO_MAGIC = b"AUD1"
AUDIO_HEADER = struct.Struct("<4sIII")  # magic, seq, sample_rate, index


class ClientType:
    HELLO = "hello"
    UTTERANCE_START = "utterance.start"
    UTTERANCE_END = "utterance.end"
    BARGE_IN = "barge_in"
    TEXT = "text"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    PLAYBACK_IDLE = "playback.idle"
    JOB_CANCEL = "job.cancel"
    CONFIG = "config"
    PING = "ping"


class ServerType:
    READY = "ready"
    STATE = "state"
    STT_PARTIAL = "stt.partial"
    STT_FINAL = "stt.final"
    USER_TEXT = "user.text"
    ASSISTANT_DELTA = "assistant.delta"
    ASSISTANT_MESSAGE = "assistant.message"
    ASSISTANT_THINKING = "assistant.thinking"
    AUDIO_START = "audio.start"
    AUDIO_END = "audio.end"
    CANCEL_AUDIO = "cancel.audio"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    CONFIRM_REQUEST = "confirm.request"
    CONFIRM_RESOLVED = "confirm.resolved"
    NOTICE = "notice"
    ERROR = "error"
    METRICS = "metrics"
    PONG = "pong"
    JOB_UPDATE = "job.update"
    ACTION_CREATED = "action.created"


class State:
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    AWAITING = "awaiting_confirmation"


def decode_client(data: str) -> dict[str, Any]:
    try:
        msg = json.loads(data)
    except (TypeError, ValueError):
        return {"type": "invalid"}
    return msg if isinstance(msg, dict) else {"type": "invalid"}


def audio_frame(seq: int, sample_rate: int, index: int, payload: bytes) -> bytes:
    return AUDIO_HEADER.pack(AUDIO_MAGIC, seq, sample_rate, index) + payload


def parse_audio_frame(frame: bytes) -> tuple[int, int, int, bytes] | None:
    if len(frame) < AUDIO_HEADER.size:
        return None
    magic, seq, sample_rate, index = AUDIO_HEADER.unpack(frame[: AUDIO_HEADER.size])
    if magic != AUDIO_MAGIC:
        return None
    return seq, sample_rate, index, frame[AUDIO_HEADER.size :]


@dataclass(slots=True)
class Config:
    """Per-session config selectable from the UI."""

    model: str = "deepseek-4.1-flash"
    voice: str = "Aiden"
    instruct: str = ""
    language: str = "English"
    personality: str = "otto"
    speed: float = 1.0
    temperature: float = 0.7
    max_tokens: int = 500
    stt_model: str = ""
    #: split work across parallel subagents (exposes the delegate_task tool)
    parallel_agents: bool = False
    #: use the "*‑background" variant of the model for subagents / background jobs
    background_model: bool = False
    #: reasoning effort for the chosen model ("" = the per-model default from config)
    reasoning_effort: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "voice": self.voice,
            "instruct": self.instruct,
            "language": self.language,
            "personality": self.personality,
            "speed": self.speed,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stt_model": self.stt_model,
            "parallel_agents": self.parallel_agents,
            "background_model": self.background_model,
            "reasoning_effort": self.reasoning_effort,
        }

    def update(self, patch: dict[str, Any]) -> None:
        for key in ("model", "voice", "instruct", "language", "personality"):
            if isinstance(patch.get(key), str):
                setattr(self, key, patch[key])
        if isinstance(patch.get("stt_model"), str) and patch["stt_model"]:
            self.stt_model = patch["stt_model"]
        for key in ("speed", "temperature"):
            if isinstance(patch.get(key), (int, float)):
                setattr(self, key, float(patch[key]))
        if isinstance(patch.get("max_tokens"), int):
            self.max_tokens = int(patch["max_tokens"])
        for key in ("parallel_agents", "background_model"):
            if isinstance(patch.get(key), bool):
                setattr(self, key, patch[key])
        if isinstance(patch.get("reasoning_effort"), str):
            self.reasoning_effort = patch["reasoning_effort"].strip()


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str = "{}"

    def args(self) -> dict[str, Any]:
        try:
            parsed = json.loads(self.arguments or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}


@dataclass(slots=True)
class ToolCalls:
    """Accumulates streamed tool-call deltas (chat-completions style)."""

    by_index: dict[int, dict[str, Any]] = field(default_factory=dict)

    def add(self, delta: dict[str, Any]) -> None:
        index = int(delta.get("index") or 0)
        slot = self.by_index.setdefault(index, {"id": "", "name": "", "arguments": ""})
        if delta.get("id"):
            slot["id"] = delta["id"]
        fn = delta.get("function") or {}
        if fn.get("name"):
            slot["name"] = fn["name"]
        if fn.get("arguments"):
            slot["arguments"] += fn["arguments"]

    def finish(self) -> list[ToolCall]:
        out: list[ToolCall] = []
        for index in sorted(self.by_index):
            slot = self.by_index[index]
            if not slot["name"]:
                continue
            out.append(
                ToolCall(
                    id=slot["id"] or f"call_{index}",
                    name=slot["name"],
                    arguments=slot["arguments"] or "{}",
                )
            )
        return out
