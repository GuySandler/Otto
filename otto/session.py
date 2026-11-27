"""The conversation harness.

One ``Session`` per websocket connection. It owns:

* turn-taking (VAD delimiters arrive from the client, transcripts from STT),
* streaming LLM text into sentence-sized TTS segments that the browser plays,
* live tool execution with modular tools, mid-turn filler speech and a
  two-step confirmation flow for irreversible actions,
* barge-in: any new speech cancels generation *and* queued audio.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .agents import SubagentRunner
from .jobs import DONE, FAILED, Job, JobManager
from .mcp import MCPManager
from .memory import EntityMemory
from .pipeline.llm import LLMError, LunarouteLLM, ReasoningDelta, StreamDone, TextDelta
from .pipeline.sentences import SentenceBuffer, clean_for_speech
from .pipeline.stt import STTError, LunarouteSTT
from .pipeline.tts import LocalTTS, TTSError
from .personality import Personality, PersonalityStore
from .protocol import (
    ClientType,
    Config,
    ServerType,
    State,
    ToolCall,
    audio_frame,
)
from .settings import Settings
from .tools.base import ToolContext, ToolResult
from .tools.registry import ToolRegistry

log = logging.getLogger("otto.session")

_APPROVE = re.compile(
    r"^(?:yes|yeah|yep|yup|sure|ok|okay|alright|all right|go ahead|go for it|do it|"
    r"send it|post it|delete it|confirm|confirmed|approve|approved|please do|yes please|"
    r"correct|affirmative|that'?s right|sounds good|perfect|absolutely)\b",
    re.I,
)
_DENY = re.compile(
    r"^(?:no|nope|nah|don'?t|do not|cancel|stop|wait|hold on|hold off|abort|negative|"
    r"forget it|never ?mind|leave it|not now)\b",
    re.I,
)


class CancelToken:
    __slots__ = ("event", "reason")

    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.reason = ""

    def cancel(self, reason: str = "") -> None:
        self.reason = reason or self.reason or "cancelled"
        self.event.set()

    @property
    def cancelled(self) -> bool:
        return self.event.is_set()


@dataclass(slots=True)
class PendingConfirmation:
    id: str
    tool_name: str
    args: dict[str, Any]
    call: ToolCall
    remaining: list[ToolCall] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    def describe(self) -> str:
        tool = self.tool_name.replace("_", " ")
        return f"{tool} ({self.id})"


class SpeechChannel:
    """Turns text into ordered, interruptible PCM segments for the browser."""

    def __init__(self, session: "Session") -> None:
        self.session = session
        self.queue: asyncio.Queue[tuple[str, int] | None] = asyncio.Queue()
        self.worker: asyncio.Task[None] | None = None
        self.next_seq = 1
        self.last_sent_seq = 0
        self.turn_first_seq: int | None = None
        self.turn_segments = 0
        self.degraded = False
        self._pending = 0
        cfg = session.settings.cfg("tts", default={}) or {}
        self.coalesce_enabled = bool(cfg.get("coalesce_queue", True))
        self.coalesce_max_chars = int(cfg.get("coalesce_max_chars", 260))
        self._carry: tuple[str, int] | None = None

    # ------------------------------------------------------------------ control
    def _ensure_worker(self) -> None:
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._run(), name="otto-speech")

    async def say(self, text: str) -> None:
        text = clean_for_speech(text)
        if not text:
            return
        self._ensure_worker()
        self._pending += 1
        await self.queue.put((text, self.next_seq))
        self.next_seq += 1

    async def drain(self) -> None:
        while self._pending > 0:
            await asyncio.sleep(0.02)

    def cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.worker = None
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
        self._pending = 0
        self._carry = None
        self.turn_segments = 0

    def begin_turn(self) -> None:
        self.turn_segments = 0
        self.turn_first_seq = self.next_seq

    def flush_seq(self) -> int:
        """Sequence id the browser should flush from (drop this and everything after)."""
        return self.turn_first_seq if self.turn_first_seq is not None else self.next_seq

    @property
    def busy(self) -> bool:
        return self._pending > 0

    # ------------------------------------------------------------------- worker
    def _coalesce(self, text: str, seq: int) -> tuple[str, int, tuple[str, int] | None]:
        """Join already-queued sentences into one TTS call.

        The first sentence still goes out immediately (that is where latency
        matters); anything already queued behind it is merged into the same
        synthesis request, which removes most of the seams that make streamed
        TTS sound choppy.
        """
        if not self.coalesce_enabled:
            return text, seq, None
        limit = self.coalesce_max_chars
        carry: tuple[str, int] | None = None
        while len(text) < limit and not self.queue.empty():
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.queue.task_done()
            if item is None:
                self._pending = max(0, self._pending - 1)
                continue
            merged = f"{text.rstrip()} {item[0].lstrip()}".strip()
            if len(merged) > limit:
                carry = item  # still counts as pending; spoken next
                break
            self._pending = max(0, self._pending - 1)
            text, seq = merged, item[1]
        return text, seq, carry

    async def _run(self) -> None:
        while True:
            if self._carry is not None:
                item, self._carry = self._carry, None
            else:
                item = await self.queue.get()
            try:
                if item is None:
                    return
                text, seq = item
                text, seq, self._carry = self._coalesce(text, seq)
                await self._speak(text, seq)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("speech segment failed")
            finally:
                self._pending = max(0, self._pending - 1)
                self.queue.task_done()

    async def _speak(self, text: str, seq: int) -> None:
        session = self.session
        config = session.config
        self.last_sent_seq = max(self.last_sent_seq, seq)
        started = time.perf_counter()
        first_chunk = True
        index = 0
        sr = 24_000
        try:
            async for chunk in session.tts.stream(
                text,
                voice=config.voice,
                instruct=config.instruct or None,
                language=config.language,
                speed=config.speed,
            ):
                if session.token is not None and session.token.cancelled:
                    return
                if first_chunk:
                    sr = chunk.sample_rate
                    await session.send(
                        {
                            "type": ServerType.AUDIO_START,
                            "seq": seq,
                            "sampleRate": sr,
                            "text": text,
                        }
                    )
                    session.metrics.setdefault("first_audio_ms", int((time.perf_counter() - started) * 1000))
                    if session.turn_started_at:
                        session.metrics.setdefault(
                            "speech_start_ms", int((time.perf_counter() - session.turn_started_at) * 1000)
                        )
                    if session.state != State.SPEAKING:
                        await session.set_state(State.SPEAKING)
                    first_chunk = False
                await session.send_bytes(audio_frame(seq, chunk.sample_rate, index, chunk.pcm))
                index += 1
        except TTSError as exc:
            if not self.degraded:
                self.degraded = True
                log.warning("tts unavailable: %s", exc)
                await session.send(
                    {
                        "type": ServerType.NOTICE,
                        "level": "warn",
                        "message": "Local voice is offline — replying in text only. Start it with ./scripts/run_tts.sh",
                    }
                )
            await session.send({"type": ServerType.AUDIO_END, "seq": seq, "ms": 0, "error": str(exc)})
            return
        if first_chunk:
            return
        self.turn_segments += 1
        await session.send(
            {
                "type": ServerType.AUDIO_END,
                "seq": seq,
                "ms": int((time.perf_counter() - started) * 1000),
            }
        )


class Session:
    def __init__(
        self,
        *,
        session_id: str,
        settings: Settings,
        llm: LunarouteLLM,
        stt: LunarouteSTT,
        tts: LocalTTS,
        registry: ToolRegistry,
        personalities: PersonalityStore,
        integrations: Any,
        config: Config,
        send_json: Callable[[dict[str, Any]], Awaitable[None]],
        send_bytes: Callable[[bytes], Awaitable[None]],
        user_id: str = "",
        actions: Any = None,
    ) -> None:
        self.id = session_id
        self.user_id = user_id
        self.actions = actions
        self.settings = settings
        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.registry = registry
        self.personalities = personalities
        self.integrations = integrations
        self.config = config
        self._send_json = send_json
        self._send_bytes = send_bytes

        self.personality: Personality = personalities.default(config.personality)
        self.system_extra: str = ""

        # cross-turn reference resolution ("move *that* to four") + detached work
        self.memory = EntityMemory()
        self.mcp = MCPManager(
            settings,
            token_provider=self._mcp_token,
            fallback_lookup=self.registry.get,
        )
        self.subagents = SubagentRunner(llm, settings, registry, integrations=integrations, mcp=self.mcp)
        self.jobs = JobManager(settings, on_event=self._on_job_event, on_done=self._on_job_done)
        self._announcements: list[Job] = []
        self._announce_task: asyncio.Task[None] | None = None

        self.messages: list[dict[str, Any]] = []
        self._build_system_message()

        self.state: str = State.IDLE
        self.token: CancelToken | None = None
        self.turn_task: asyncio.Task[None] | None = None
        self.turn_index = 0
        self.pending: PendingConfirmation | None = None
        self.speech = SpeechChannel(self)
        self.metrics: dict[str, Any] = {}
        self.turn_started_at: float | None = None
        self.playback_idle_seq = 0
        self._hedge_task: asyncio.Task[None] | None = None

        # capture state
        self.audio = bytearray()
        self.last_audio_at = 0.0
        self.watchdog_task: asyncio.Task[None] | None = None
        self.utterance_open = False
        self.utterance_started = 0.0
        self.partial_task: asyncio.Task[None] | None = None
        self._partial_last_len = 0
        self._partial_text = ""
        self._closed = False

        self.tool_context = ToolContext(
            settings=settings,
            integrations=integrations,
            session=self,
        )

    # ------------------------------------------------------------------ plumbing
    async def send(self, payload: dict[str, Any]) -> None:
        payload.setdefault("type", "notice")
        await self._send_json(payload)

    async def send_bytes(self, data: bytes) -> None:
        await self._send_bytes(data)

    async def set_state(self, state: str, detail: str | None = None) -> None:
        if self.state == state and detail is None:
            return
        self.state = state
        await self.send({"type": ServerType.STATE, "state": state, "detail": detail})

    # ------------------------------------------------------------------ prompts
    def _user_name_block(self) -> str:
        """The user's own name (cached by the people_profile tool) for sign-offs."""
        try:
            integrations = self.integrations
            users = getattr(integrations, "users", None)
            user_id = getattr(integrations, "user_id", None)
            if users is not None and user_id:
                name = str(users.get_secret(user_id, "profile_name") or "").strip()
                if name:
                    return (
                        f" The user's name is {name}; sign drafted email with it "
                        "('Best, <name>') unless the conversation says otherwise."
                    )
        except Exception:  # noqa: BLE001 — a missing name must never break the prompt
            pass
        return ""

    def _build_system_message(self, *, replace: bool = False) -> None:
        global_rules = str(self.settings.cfg("session", "global_rules", default="") or "")
        memory_block = self.memory.prompt_block()
        user_name = self._user_name_block()
        send_tool = self.registry.get("gmail_send") if hasattr(self, "registry") else None
        try:
            send_available = bool(send_tool and send_tool.is_available(self.tool_context))
        except Exception:  # noqa: BLE001
            send_available = False
        mcp_note = self.mcp.prompt_note(send_available=send_available) if hasattr(self, "mcp") else ""
        extra = "\n\n".join(part for part in (self.system_extra, mcp_note, user_name, memory_block) if part)
        prompt = self.personality.build_system_prompt(global_rules=global_rules, extra=extra)
        if self.messages and self.messages[0].get("role") == "system" and replace:
            self.messages[0] = {"role": "system", "content": prompt}
        elif not self.messages:
            self.messages.insert(0, {"role": "system", "content": prompt})
        else:
            self.messages[0] = {"role": "system", "content": prompt}

    async def _mcp_token(self) -> str:
        """Access token handed to the MCP servers (this user's Google account)."""
        google = self.integrations.google
        if not google.connected:
            raise RuntimeError("Google is not connected for this account")
        return await google.access_token()

    def flags(self) -> dict[str, bool]:
        return {
            "parallel_agents": bool(self.config.parallel_agents),
        }

    def agent_model(self) -> str:
        """Model used for subagents and background jobs."""
        if self.config.background_model:
            return self.llm.background_variant(self.config.model)
        return self.config.model

    def _humanize_ids(self, text: str) -> str:
        """Raw ids are for machines — speak the remembered title instead.

        'I'll update ku5mdqp4… Okay?' is unacceptable when the conversation
        already saw the event; EntityMemory knows the mapping.
        """
        memory = getattr(self, "memory", None)
        if not text or memory is None:
            return text
        for entity in sorted(memory.items, key=lambda item: len(item.id), reverse=True):
            if entity.id and len(entity.id) >= 6 and entity.id in text:
                text = text.replace(entity.id, f"'{entity.label}'")
        return text

    def _enrich_preview(self, preview: Any, args: dict[str, Any]) -> Any:
        """Fill a confirmation card's title from memory when args only carry an id."""
        if not isinstance(preview, dict) or preview.get("title"):
            return preview
        memory = getattr(self, "memory", None)
        if memory is None:
            return preview
        wanted = {str(args.get(key) or "") for key in ("eventId", "event_id", "id", "messageId", "threadId")}
        for entity in memory.items:
            if entity.id and entity.id in wanted:
                return {**preview, "title": entity.label}
        return preview

    def _llm_extra(self) -> dict[str, Any]:
        """Per-session request overrides — the Thinking selector in the UI."""
        effort = (getattr(self.config, "reasoning_effort", "") or "").strip()
        return {"reasoning_effort": effort} if effort else {}

    def apply_config(self, patch: dict[str, Any]) -> None:
        previous_personality = self.config.personality
        self.config.update(patch)
        # always reload the persona file when one is named: the UI re-sends the
        # config after editing a prompt, and that edit must take effect now
        if self.config.personality != previous_personality or "personality" in patch:
            try:
                self.personality = self.personalities.load(self.config.personality, use_cache=False)
            except KeyError:
                self.personality = self.personalities.default()
                self.config.personality = self.personality.id
            self._build_system_message(replace=True)
        if "voice" in patch:
            self.config.voice = patch["voice"]

    # ------------------------------------------------------------------ ws entry
    async def handle_message(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == ClientType.HELLO:
            await self.send(
                {
                    "type": ServerType.READY,
                    "session": self.id,
                    "config": self.config.to_dict(),
                    "personality": self.personality.to_dict(),
                    "state": self.state,
                    "vad": {
                        "silenceMs": int(self.settings.cfg("turn_taking", "endpoint_silence_ms", default=550)),
                    },
                }
            )
            return
        if kind == ClientType.PING:
            await self.send({"type": ServerType.PONG})
            return
        if kind == ClientType.UTTERANCE_START:
            await self.on_utterance_start()
            return
        if kind == ClientType.UTTERANCE_END:
            await self.on_utterance_end(msg.get("durationMs"))
            return
        if kind == ClientType.BARGE_IN:
            await self.on_barge_in()
            return
        if kind == ClientType.TEXT:
            text = str(msg.get("text") or "").strip()
            if text:
                await self.send({"type": ServerType.USER_TEXT, "text": text, "source": "typed"})
                if self.pending is not None:
                    await self._handle_confirmation_answer(text, source="typed")
                else:
                    await self._start_turn(user_text=text, source="typed")
            return
        if kind == ClientType.CONFIRM:
            pend_id = str(msg.get("id") or "")
            approved = bool(msg.get("approved"))
            if self.pending is not None and (not pend_id or pend_id == self.pending.id):
                await self._resolve_confirmation(approved, source="button")
            return
        if kind == ClientType.CANCEL:
            await self._cancel_turn(notify=True)
            await self.set_state(State.LISTENING)
            return
        if kind == ClientType.PLAYBACK_IDLE:
            try:
                self.playback_idle_seq = max(self.playback_idle_seq, int(msg.get("seq") or 0))
            except (TypeError, ValueError):
                pass
            # while the model is still producing we stay "speaking" (a momentary
            # gap between sentences must not flicker the UI back to idle)
            if self.turn_task is not None and not self.turn_task.done():
                return
            if self.state in (State.SPEAKING, State.THINKING) and self._playback_settled():
                await self.set_state(State.IDLE)
            return
        if kind == ClientType.JOB_CANCEL:
            job_id = str(msg.get("id") or "")
            cancelled = self.jobs.cancel(job_id)
            await self.send({"type": "job.cancelled", "id": job_id, "ok": cancelled})
            return
        if kind == ClientType.CONFIG:
            self.apply_config(msg)
            await self.send({"type": "config", "config": self.config.to_dict()})
            return
        log.debug("unhandled client message: %s", kind)

    # -------------------------------------------------------------------- audio
    async def on_audio(self, chunk: bytes) -> None:
        if self._closed or not chunk or not self.utterance_open:
            return
        self.last_audio_at = time.perf_counter()
        self.audio.extend(chunk)
        cap = float(self.settings.cfg("stt", "max_utterance_s", default=60)) * 16_000 * 2
        if len(self.audio) > cap:
            del self.audio[: len(self.audio) - int(cap)]

    async def on_utterance_start(self) -> None:
        self.audio.clear()
        self.utterance_open = True
        self.utterance_started = time.perf_counter()
        self._partial_last_len = 0
        self._partial_text = ""
        if self.settings.cfg("stt", "partials", default=True):
            self._start_partials()
        self._start_watchdog()

    async def on_utterance_end(self, duration_ms: int | None = None) -> None:
        self.utterance_open = False
        await self._stop_partials()
        await self._stop_watchdog()
        pcm = bytes(self.audio)
        self.audio.clear()
        min_bytes = int(float(self.settings.cfg("stt", "min_utterance_ms", default=220)) * 16)
        if len(pcm) < min_bytes:
            return
        await self.set_state(State.LISTENING, detail="transcribing")
        started = time.perf_counter()
        try:
            transcript = await self.stt.transcribe(pcm, model=self.config.stt_model or None)
        except STTError as exc:
            await self.send({"type": ServerType.ERROR, "message": f"Transcription failed: {exc}"})
            # one spoken apology beats a silent failure on a voice turn — the
            # user is left hanging otherwise
            try:
                await self.speech.say(
                    "Sorry, the speech service just hiccuped and I lost that. Could you say it again?"
                )
                await self.speech.drain()
            except Exception:  # noqa: BLE001 — the apology must never mask the error
                pass
            await self.set_state(State.IDLE)
            return
        text = transcript.text.strip()
        self.metrics["stt_ms"] = int((time.perf_counter() - started) * 1000)
        self.metrics["stt_model"] = self.config.stt_model or self.settings.stt_model
        await self.send(
            {
                "type": ServerType.STT_FINAL,
                "text": text,
                "durationMs": duration_ms or transcript.duration_ms,
                "latencyMs": self.metrics["stt_ms"],
            }
        )
        if not text:
            await self.set_state(State.IDLE)
            return
        await self.send({"type": ServerType.USER_TEXT, "text": text, "source": "voice"})
        if self.pending is not None:
            await self._handle_confirmation_answer(text, source="voice")
        else:
            await self._start_turn(user_text=text, source="voice")

    # The client decides when a turn ends, but it must never be able to hang one.
    def _start_watchdog(self) -> None:
        if self.watchdog_task is not None:
            self.watchdog_task.cancel()
        self.watchdog_task = asyncio.create_task(self._watch_utterance(), name="otto-utterance-watchdog")

    async def _stop_watchdog(self) -> None:
        task = self.watchdog_task
        self.watchdog_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _watch_utterance(self) -> None:
        """Force-end an utterance the client forgot to close.

        A VAD that never settles (a hissy room, AGC, a stuck hold) used to leave the
        conversation silent forever: partial transcriptions kept flowing and no turn
        was ever started. Now the server bounds it: too long overall, or no audio for
        a while, and we transcribe what we have and run the turn.
        """
        max_s = float(self.settings.cfg("stt", "max_utterance_s", default=60))
        stall_s = float(self.settings.cfg("turn_taking", "utterance_stall_s", default=6))
        try:
            while self.utterance_open and not self._closed:
                await asyncio.sleep(1.0)
                if not self.utterance_open:
                    return
                now = time.perf_counter()
                open_for = now - self.utterance_started
                quiet_for = now - (self.last_audio_at or self.utterance_started)
                reason = ""
                if open_for >= max_s:
                    reason = f"open for {open_for:.0f}s (limit {max_s:.0f}s)"
                elif quiet_for >= stall_s:
                    reason = f"no audio for {quiet_for:.1f}s (limit {stall_s:.0f}s)"
                if not reason:
                    continue
                if not self.audio:
                    self.utterance_open = False
                    return
                log.warning("utterance watchdog: force-ending — %s", reason)
                await self.send(
                    {
                        "type": ServerType.NOTICE,
                        "level": "warn",
                        "message": "The microphone stayed open — closing the turn so I can answer.",
                    }
                )
                await self.on_utterance_end()
                return
        except asyncio.CancelledError:
            raise

    def _start_partials(self) -> None:
        if self.partial_task is not None:
            self.partial_task.cancel()
        self.partial_task = asyncio.create_task(self._partial_loop(), name="otto-partials")

    async def _stop_partials(self) -> None:
        task = self.partial_task
        self.partial_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _partial_loop(self) -> None:
        interval = float(self.settings.cfg("stt", "partial_interval_ms", default=550)) / 1000
        max_s = float(self.settings.cfg("stt", "max_utterance_s", default=60))
        calls = 0
        call_budget = max(40, int(max_s / max(interval, 0.3)))  # captions for the whole turn, however long
        try:
            while True:
                await asyncio.sleep(interval)
                calls += 1
                if calls > call_budget or (time.perf_counter() - self.utterance_started) > max_s:
                    return  # long monologue: stop paying for interim transcripts
                if not self.utterance_open or len(self.audio) < 16_000:
                    continue
                if len(self.audio) - self._partial_last_len < 4000:
                    continue
                self._partial_last_len = len(self.audio)
                snapshot = bytes(self.audio)
                try:
                    interim = await asyncio.wait_for(self.stt.transcribe(snapshot), timeout=6.0)
                except (asyncio.TimeoutError, STTError):
                    continue
                text = interim.text.strip()
                if text and text != self._partial_text:
                    self._partial_text = text
                    await self.send({"type": ServerType.STT_PARTIAL, "text": text})
        except asyncio.CancelledError:
            raise

    # -------------------------------------------------------------------- turns
    async def _turn_tools(self) -> list[dict[str, Any]]:
        """Tool schemas for this turn: MCP tools replace the REST ones by name."""
        mcp_tools = await self.mcp.discover()
        usable = [tool for tool in mcp_tools if tool.is_available(self.tool_context)]
        mcp_names = {tool.name for tool in usable}
        blocked_categories = {"gmail", "calendar"} if self.mcp.authoritative else set()
        if blocked_categories:
            log.info("mcp: strict mode — Gmail/Calendar run on MCP only")
        schemas = self.registry.schemas(
            flags=self.flags(),
            ctx=self.tool_context,
            exclude=mcp_names,
            exclude_categories=blocked_categories,
        )
        schemas.extend(tool.to_schema() for tool in usable)
        for notice in self.mcp.drain_notices():
            await self.send({"type": ServerType.NOTICE, "level": "warn", "message": notice})
        return schemas

    async def _cancel_turn(self, *, notify: bool = True, reason: str = "cancelled") -> None:
        task = self.turn_task
        self.turn_task = None
        if task is not None and not task.done():
            if self.token is not None:
                self.token.cancel(reason)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._cancel_hedge()
        flush_from = self.speech.flush_seq()
        self.speech.cancel()
        if notify:
            await self.send({"type": ServerType.CANCEL_AUDIO, "fromSeq": flush_from})
        self.token = None

    async def _start_turn(self, *, user_text: str | None = None, source: str = "voice") -> None:
        flush = self.speech.busy or self.state in (State.SPEAKING, State.THINKING)
        await self._cancel_turn(notify=flush, reason="new turn")
        if user_text:
            self.messages.append({"role": "user", "content": user_text})
            self.trim_history()
        self._build_system_message(replace=True)
        self.turn_index += 1
        token = CancelToken()
        self.token = token
        self.metrics = {"turn": self.turn_index, "source": source}
        self.speech.begin_turn()
        self.turn_task = asyncio.create_task(self._run_turn(token), name=f"otto-turn-{self.turn_index}")

    async def _run_turn(self, token: CancelToken) -> None:
        started = time.perf_counter()
        self.turn_started_at = started
        turn_cfg = self.settings.cfg("turn_taking", default={}) or {}
        max_iters = int(turn_cfg.get("max_tool_iterations", 6))
        first_max = int(self.settings.cfg("tts", "first_chunk_max_chars", default=60))
        max_chars = int(self.settings.cfg("tts", "max_chunk_chars", default=220))
        min_chars = int(self.settings.cfg("tts", "min_chunk_chars", default=0))
        thinking_sent = False
        try:
            await self.set_state(State.THINKING)
            for iteration in range(max_iters + 1):
                if token.cancelled:
                    return
                tools = await self._turn_tools()
                buffer = SentenceBuffer(first_max_chars=first_max, max_chars=max_chars, min_chars=min_chars)
                text_parts: list[str] = []
                tool_calls: list[ToolCall] = []
                message_id = f"m{self.turn_index}.{iteration}"
                llm_ttft: int | None = None
                flags = {"text": False}
                hedge_ms = float(turn_cfg.get("first_token_filler_ms", 0) or 0)
                hedge = (
                    asyncio.create_task(self._hedge(token, flags, hedge_ms / 1000))
                    if hedge_ms > 0
                    else None
                )
                self._hedge_task = hedge
                stream = self.llm.stream(
                    self.messages,
                    model=self.config.model,
                    tools=tools or None,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    extra_body=self._llm_extra(),
                )
                try:
                    async for event in stream:
                        if token.cancelled:
                            await stream.aclose()
                            return
                        if isinstance(event, TextDelta):
                            if not flags["text"]:
                                flags["text"] = True
                                if hedge is not None and not hedge.done():
                                    hedge.cancel()
                            if llm_ttft is None:
                                llm_ttft = int((time.perf_counter() - started) * 1000)
                                self.metrics["llm_ttft_ms"] = llm_ttft
                            text_parts.append(event.text)
                            await self.send(
                                {"type": ServerType.ASSISTANT_DELTA, "id": message_id, "text": event.text}
                            )
                            for sentence in buffer.feed(event.text):
                                await self.speech.say(sentence)
                        elif isinstance(event, ReasoningDelta):
                            if not thinking_sent:
                                thinking_sent = True
                                await self.send({"type": ServerType.ASSISTANT_THINKING, "id": message_id})
                        elif isinstance(event, StreamDone):
                            tool_calls = event.tool_calls
                            if event.usage:
                                self.metrics["usage"] = event.usage
                except LLMError as exc:
                    await self.send({"type": ServerType.ERROR, "message": str(exc)})
                    await self.set_state(State.IDLE)
                    return

                if hedge is not None and not hedge.done():
                    hedge.cancel()
                for sentence in buffer.flush():
                    await self.speech.say(sentence)
                full_text = "".join(text_parts).strip()
                if full_text:
                    await self.send(
                        {
                            "type": ServerType.ASSISTANT_MESSAGE,
                            "id": message_id,
                            "text": full_text,
                            "spoken": not self.speech.degraded,
                        }
                    )

                if not tool_calls:
                    if full_text:
                        self.messages.append({"role": "assistant", "content": full_text})
                    break

                self.messages.append(
                    {
                        "role": "assistant",
                        "content": full_text or None,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {"name": call.name, "arguments": call.arguments},
                            }
                            for call in tool_calls
                        ],
                    }
                )
                outcome = await self._execute_calls(token, tool_calls)
                if outcome == "paused":
                    self._emit_metrics(started)
                    return
                if token.cancelled:
                    return
            await self.speech.drain()
            if token.cancelled:
                return
            self._emit_metrics(started)
            if self.state != State.AWAITING:
                # the browser owns the end of the turn: it reports playback.idle
                # once the last scheduled buffer has actually been heard
                settled = self._playback_settled()
                if self.speech.turn_segments and not settled:
                    await self.set_state(State.SPEAKING)
                else:
                    await self.set_state(State.IDLE)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("turn failed")
            await self.send({"type": ServerType.ERROR, "message": f"{type(exc).__name__}: {exc}"})
            await self.set_state(State.IDLE)

    # --------------------------------------------------------------------- jobs
    def _on_job_event(self, job: Job) -> None:
        asyncio.create_task(self.send({"type": ServerType.JOB_UPDATE, **job.to_event()}))

    def _on_job_done(self, job: Job) -> None:
        self._announcements.append(job)
        if self._announce_task is None or self._announce_task.done():
            self._announce_task = asyncio.create_task(self._announce_loop(), name="otto-announce")

    async def _announce_loop(self) -> None:
        """Speak finished background work — but never on top of the user."""
        waited = 0.0
        while self._announcements and not self._closed:
            busy = (self.turn_task is not None and not self.turn_task.done()) or self.pending is not None
            if busy:
                await asyncio.sleep(0.4)
                waited += 0.4
                if waited > 300:
                    self._announcements.clear()
                    return
                continue
            job = self._announcements.pop(0)
            waited = 0.0
            await self._announce_job(job)

    async def _announce_job(self, job: Job) -> None:
        if job.status == FAILED:
            note = (
                f'[background task failed] "{job.title}" failed: {job.error}. '
                "Tell the user briefly that it did not work, in one sentence."
            )
        else:
            note = (
                f'[background task finished] "{job.title}":\n{job.result[:1500]}\n\n'
                "Tell the user in one or two short spoken sentences what you found, and offer the "
                "details if they want them. This is your own background work, not a user request."
            )
        self.messages.append({"role": "user", "content": note})
        await self._start_turn(user_text=None, source="job")
        task = self.turn_task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    def _playback_settled(self) -> bool:
        """True when the browser has finished every audio segment we sent."""
        return self.playback_idle_seq >= self.speech.last_sent_seq

    def _cancel_hedge(self) -> None:
        task = self._hedge_task
        self._hedge_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _hedge(self, token: CancelToken, flags: dict[str, bool], delay: float) -> None:
        """Say a short human noise while the model is slow to start talking.

        Repeats a couple of times with growing gaps so a gateway hiccup in the
        multi-second range still sounds like a person thinking, not a dead line.
        """
        spoke = 0
        try:
            while spoke < 3 and not token.cancelled and not flags["text"]:
                await asyncio.sleep(delay if spoke == 0 else delay * 2.4)
                if token.cancelled or flags["text"] or self.pending is not None:
                    return
                if self.speech.busy or self.speech.turn_segments > 0 or self.speech.degraded:
                    return
                fillers = self.personality.thinking_fillers
                if not fillers:
                    return
                await self.speech.say(random.choice(fillers))
                await self.set_state(State.SPEAKING)
                spoke += 1
        except asyncio.CancelledError:
            raise

    def _emit_metrics(self, started: float) -> None:
        self.metrics["turn_ms"] = int((time.perf_counter() - started) * 1000)
        asyncio.create_task(self.send({"type": ServerType.METRICS, "latency": dict(self.metrics)}))

    # -------------------------------------------------------------------- tools
    async def _execute_calls(self, token: CancelToken, calls: list[ToolCall]) -> str:
        for position, call in enumerate(calls):
            if token.cancelled:
                return "cancelled"
            self._cancel_hedge()
            tool = self.mcp.get(call.name) or self.registry.get(call.name)
            args = call.args()
            await self.send(
                {
                    "type": ServerType.TOOL_CALL,
                    "id": call.id,
                    "name": call.name,
                    "args": args,
                    "category": getattr(tool, "category", "unknown"),
                }
            )
            if tool is None:
                result = ToolResult(ok=False, error=f"Unknown tool '{call.name}'", summary="That tool does not exist.")
                await self._emit_tool_result(call, result, ms=0)
                self.messages.append(self._tool_message(call, result))
                continue

            if tool.needs_confirmation(args):
                self.pending = PendingConfirmation(
                    id=f"cf_{uuid.uuid4().hex[:8]}",
                    tool_name=call.name,
                    args=args,
                    call=call,
                    remaining=calls[position + 1 :],
                )
                summary = self._humanize_ids(tool.confirmation_text(args))
                preview = self._enrich_preview(tool.confirmation_preview(args), args)
                await self.send(
                    {
                        "type": ServerType.CONFIRM_REQUEST,
                        "id": self.pending.id,
                        "tool": call.name,
                        "name": tool.name,
                        "args": args,
                        "summary": summary,
                        "preview": preview,
                        "danger": bool(tool.danger),
                    }
                )
                await self.speech.say(summary)
                await self.speech.drain()
                await self.set_state(State.AWAITING)
                return "paused"

            result, ms = await self._run_tool_with_filler(token, tool, args)
            if token.cancelled:
                return "cancelled"
            await self._emit_tool_result(call, result, ms=ms)
            self.messages.append(self._tool_message(call, result))
        return "done"

    async def _run_tool_with_filler(self, token: CancelToken, tool, args: dict[str, Any]) -> tuple[ToolResult, int]:
        started = time.perf_counter()
        task = asyncio.create_task(tool.invoke(self.tool_context, args))
        delay = float(self.settings.cfg("turn_taking", "tool_filler_delay_ms", default=250)) / 1000
        try:
            done, _ = await asyncio.wait({task}, timeout=max(0.0, delay))
            if not done and not self.speech.busy and self.speech.turn_segments == 0 and not token.cancelled:
                filler = self.personality.tool_filler
                if filler:
                    await self.speech.say(filler)
            await self.set_state(State.THINKING, detail=f"tool:{tool.name}")
            result = await task
        except asyncio.CancelledError:
            task.cancel()
            raise
        ms = int((time.perf_counter() - started) * 1000)
        return result, ms

    async def _emit_tool_result(self, call: ToolCall, result: ToolResult, *, ms: int) -> None:
        try:
            self.memory.remember_tool_display(result.display)
        except Exception:  # noqa: BLE001 - memory must never break a turn
            log.exception("memory update failed")
        await self._record_actions(call, result)
        await self.send(
            {
                "type": ServerType.TOOL_RESULT,
                "id": call.id,
                "name": call.name,
                "ok": result.ok,
                "summary": result.summary[:1200],
                "display": result.display,
                "error": result.error,
                "ms": ms,
            }
        )

    async def _record_actions(self, call: ToolCall, result: ToolResult) -> None:
        """Persist anything the user has to finish themselves (e.g. a draft)."""
        if self.actions is None or not self.user_id or not result.ok:
            return
        from .actions import actions_from_display

        email = ""
        try:
            email = self.integrations.google.account_email
        except Exception:  # noqa: BLE001
            email = ""
        try:
            drafts = actions_from_display(result.display, account_email=email)
        except Exception:  # noqa: BLE001
            log.exception("could not derive actions from %s", call.name)
            return
        for spec in drafts:
            status = spec.pop("status_override", None)
            spec.pop("detail_ref", None)
            try:
                action = self.actions.record(
                    self.user_id,
                    kind=spec.pop("kind"),
                    title=spec.pop("title"),
                    subtitle=spec.pop("subtitle", ""),
                    detail=spec.pop("detail", ""),
                    url=spec.pop("url", ""),
                    ref=spec.pop("ref", ""),
                    source=call.name,
                    payload={"tool": call.name, "args": call.args()},
                    **({"status": status} if status else {}),
                )
            except Exception:  # noqa: BLE001
                log.exception("could not persist an action")
                continue
            if action.status != "open":
                continue
            await self.send(
                {
                    "type": ServerType.ACTION_CREATED,
                    "action": action.to_dict(),
                    "open": self.actions.count(self.user_id),
                }
            )

    @staticmethod
    def _tool_message(call: ToolCall, result: ToolResult) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": result.to_model_payload(),
        }

    # ------------------------------------------------------------ confirmation
    async def _handle_confirmation_answer(self, text: str, *, source: str) -> None:
        intent = await self._classify_answer(text)
        if intent == "approve":
            await self._resolve_confirmation(True, source=source)
        elif intent == "deny":
            await self._resolve_confirmation(False, source=source)
        else:
            # the user moved on: drop the pending action and treat it as a new turn
            await self._drop_pending(reason="user moved on")
            await self._start_turn(user_text=text, source=source)

    async def _classify_answer(self, text: str) -> str:
        cleaned = text.strip().lower()
        if not cleaned:
            return "other"
        if _APPROVE.match(cleaned):
            return "approve"
        if _DENY.match(cleaned):
            return "deny"
        words = cleaned.split()
        if len(words) > 7:
            return "other"
        prompt = (
            "The assistant asked the user to confirm an action. The user replied: "
            f'"{text}". Reply with exactly one word: APPROVE if they agreed, DENY if they '
            "refused, or OTHER if they said something unrelated."
        )
        try:
            answer = await self.llm.complete(
                [{"role": "user", "content": prompt}],
                model=self.config.model,
                max_tokens=6,
                temperature=0.0,
            )
        except LLMError:
            return "other"
        answer = answer.strip().upper()
        if answer.startswith("APPROVE"):
            return "approve"
        if answer.startswith("DENY"):
            return "deny"
        return "other"

    async def _resolve_confirmation(self, approved: bool, *, source: str) -> None:
        pend = self.pending
        if pend is None:
            return
        self.pending = None
        await self.send({"type": ServerType.CONFIRM_RESOLVED, "id": pend.id, "approved": approved})
        await self._cancel_turn(notify=True, reason="confirmation resolved")
        token = CancelToken()
        self.token = token
        self.speech.begin_turn()
        self.metrics = {"turn": self.turn_index, "source": source, "confirmation": True}
        self.turn_task = asyncio.create_task(
            self._finish_confirmation(pend, approved, token), name=f"otto-confirm-{self.turn_index}"
        )

    async def _finish_confirmation(self, pend: PendingConfirmation, approved: bool, token: CancelToken) -> None:
        """Run (or refuse) the confirmed tool, then let the model react to it."""
        # resolve exactly like the tool-call loop: MCP tools live in the MCP
        # manager, not the local registry — a registry-only lookup turned every
        # approved MCP action into "Unknown tool" and nothing ran
        tool = self.mcp.get(pend.tool_name) or self.registry.get(pend.tool_name)
        if approved and tool is not None:
            result, ms = await self._run_tool_with_filler(token, tool, pend.args)
            await self._emit_tool_result(pend.call, result, ms=ms)
            self.messages.append(self._tool_message(pend.call, result))
            if pend.remaining:
                outcome = await self._execute_calls(token, pend.remaining)
                if outcome == "paused":
                    return
        else:
            note = "The user declined." if tool is not None else f"Unknown tool {pend.tool_name}."
            for call in [pend.call] + list(pend.remaining):
                result = ToolResult(ok=False, error="user_declined", summary=note)
                await self._emit_tool_result(call, result, ms=0)
                self.messages.append(self._tool_message(call, result))
        await self._run_turn(token)

    async def _drop_pending(self, *, reason: str) -> None:
        pend = self.pending
        self.pending = None
        if pend is None:
            return
        await self.send({"type": ServerType.CONFIRM_RESOLVED, "id": pend.id, "approved": False})
        for call in [pend.call] + list(pend.remaining):
            result = ToolResult(ok=False, error="user_declined", summary=f"Cancelled: {reason}.")
            self.messages.append(self._tool_message(call, result))

    # ---------------------------------------------------------------- barge-in
    async def on_barge_in(self) -> None:
        if self.pending is not None:
            # the user is probably answering the confirmation question
            flush_from = self.speech.flush_seq()
            self.speech.cancel()
            await self.send({"type": ServerType.CANCEL_AUDIO, "fromSeq": flush_from})
            await self.set_state(State.LISTENING)
            return
        if self.turn_task is None or self.turn_task.done():
            await self.set_state(State.LISTENING)
            return
        await self._cancel_turn(notify=True, reason="barge-in")
        await self.set_state(State.LISTENING)

    # ------------------------------------------------------------------ close
    async def close(self) -> None:
        self._closed = True
        await self.mcp.aclose()
        await self.jobs.aclose()
        if self._announce_task is not None:
            self._announce_task.cancel()
        await self._stop_partials()
        await self._stop_watchdog()
        await self._cancel_turn(notify=False, reason="session closed")

    # ------------------------------------------------------------------ helpers
    def history_snapshot(self) -> list[dict[str, Any]]:
        return [dict(m) for m in self.messages]

    def trim_history(self) -> None:
        limit = int(self.settings.cfg("turn_taking", "history_turns", default=24)) * 2 + 1
        if len(self.messages) <= limit:
            return
        keep_from = len(self.messages) - limit
        # never start on a tool message: walk back to its assistant tool_calls turn
        while keep_from < len(self.messages) and self.messages[keep_from].get("role") == "tool":
            keep_from -= 1
        self.messages = [self.messages[0]] + self.messages[max(1, keep_from) :]
