"""Otto's local Qwen3-TTS streaming service (pinned to GPU1).

The Otto server talks to this process over the small JSON API documented in
``tts/README.md``: ``/health`` + ``/voices`` for discovery, ``POST /speak`` for
raw PCM16LE mono audio streamed while the model generates it (see
``otto/pipeline/tts.py`` for the client side), ``GET /sample`` for a complete
WAV used by the UI voice preview, and ``POST /warmup``.

Only one inference runs at a time (one GPU): every job holds ``Engine.lock``
until its worker thread really exits, so a client that disconnects mid-utterance
stops generation at the next chunk instead of overlapping the next request.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import math
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compat  # noqa: E402  - must be applied before faster_qwen3_tts is imported

log = logging.getLogger("otto.tts.server")

SAMPLE_RATE = 24_000
# 2048 int16 samples per HTTP chunk: enough to keep framing overhead small
# without ever delaying the first audio beyond one model chunk.
PCM_CHUNK_BYTES = 2048 * 2
DISCONNECT_POLL_S = 0.25

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_INSTRUCT = (
    "Speak like a warm, relaxed friend in a spoken conversation. Keep a calm, "
    "even pace: do not rush or speed up toward the end of long sentences, and "
    "hold the same energy and tone from one sentence to the next, breathing "
    "briefly between sentences the way a person naturally would."
)
DEFAULT_SAMPLE_TEXT = "Hi, I'm Otto. This is how I sound when we talk."
DEFAULT_VOICE = "Aiden"
CLONE_VOICE_ID = "cloned"
DESIGN_VOICE_ID = "design"

SUPPORTED_LANGUAGES = (
    "Chinese",
    "English",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
    "Beijing_Dialect",
    "Sichuan_Dialect",
)

PRESET_VOICES: tuple[dict[str, str], ...] = (
    {"id": "Aiden", "description": "Sunny American male, clear midrange",
     "language": "English", "kind": "preset"},
    {"id": "Ryan", "description": "Dynamic male, strong rhythmic drive",
     "language": "English", "kind": "preset"},
    {"id": "Vivian", "description": "Bright young female",
     "language": "Chinese", "kind": "preset"},
    {"id": "Serena", "description": "Warm gentle female",
     "language": "Chinese", "kind": "preset"},
    {"id": "Uncle_Fu", "description": "Low mellow male",
     "language": "Chinese", "kind": "preset"},
    {"id": "Dylan", "description": "Youthful Beijing male",
     "language": "Chinese (Beijing)", "kind": "preset"},
    {"id": "Eric", "description": "Chengdu male",
     "language": "Chinese (Sichuan)", "kind": "preset"},
    {"id": "Ono_Anna", "description": "Playful Japanese female",
     "language": "Japanese", "kind": "preset"},
    {"id": "Sohee", "description": "Warm Korean female",
     "language": "Korean", "kind": "preset"},
)

# Set from the lifespan shutdown hook so worker threads stop even if a request
# is still streaming when SIGINT arrives.
SHUTDOWN = threading.Event()


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Config:
    """Service configuration; every field is settable through OTTO_TTS_* env vars."""

    #: "chatterbox" (ResembleAI, default) or "qwen" (faster-qwen3-tts)
    backend: str = "chatterbox"
    model: str = DEFAULT_MODEL
    device: str = "cuda:0"
    #: chatterbox: reference clips for zero-shot cloning (voices/<name>.wav)
    voices_dir: str = "voices"
    #: chatterbox: 0.25 = flat reading, 1.0 = very expressive
    exaggeration: float = 0.7
    #: chatterbox: Resemble's perth watermark adds a faint burst at clip starts;
    #: a private assistant has no use for it
    watermark: bool = False
    #: linear fade-in applied to the first samples of every segment
    fade_in_ms: int = 12
    #: chatterbox conditioning strength — high values make delivery *stiff*,
    #: low values let prosody move; pace is handled by `stretch` instead
    cfg_weight: float = 0.5
    #: post-generation time-stretch applied to every clip (1.12 ≈ 12% slower).
    #: Pitch-preserving WSOLA — the model's pace knobs barely move a clone, and
    #: cranking cfg up kills the intonation, so the audio is stretched instead.
    stretch: float = 1.12
    chunk_size: int = 4
    instruct: str = DEFAULT_INSTRUCT
    mock: bool = False
    ref_audio: str | None = None
    ref_text: str | None = None
    max_chars: int = 400
    trim_silence: bool = True
    lead_keep_ms: int = 12
    #: a fixed, identical breath after every segment — the model's own tail is
    #: trimmed away and replaced, so cadence stays even across seams. This must
    #: sit at or above the model's own in-clip sentence pauses, or clip seams
    #: sound pinched next to in-clip pauses and transitions stumble.
    pause_ms: int = 280
    tail_keep_ms: int = 25
    tail_hold_ms: int = 150

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            backend=(os.environ.get("OTTO_TTS_BACKEND") or "chatterbox").strip().lower(),
            model=os.environ.get("OTTO_TTS_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            device=os.environ.get("OTTO_TTS_DEVICE", "cuda:0").strip() or "cuda:0",
            chunk_size=max(1, _env_int("OTTO_TTS_CHUNK_SIZE", 4)),
            voices_dir=os.environ.get("OTTO_TTS_VOICES_DIR", "voices").strip() or "voices",
            exaggeration=_env_float("OTTO_TTS_EXAGGERATION", 0.7, 0.05, 1.0),
            watermark=os.environ.get("OTTO_TTS_WATERMARK", "0").strip().lower()
            in {"1", "true", "yes", "on"},
            fade_in_ms=max(0, _env_int("OTTO_TTS_FADE_IN_MS", 12)),
            cfg_weight=_env_float("OTTO_TTS_CFG", 0.5, 0.3, 0.8),
            stretch=_env_float("OTTO_TTS_STRETCH", 1.12, 0.5, 2.0),
            instruct=os.environ.get("OTTO_TTS_INSTRUCT", DEFAULT_INSTRUCT),
            mock=os.environ.get("OTTO_TTS_MOCK", "0").strip().lower() in {"1", "true", "yes", "on"},
            ref_audio=os.environ.get("OTTO_TTS_REF_AUDIO") or None,
            ref_text=os.environ.get("OTTO_TTS_REF_TEXT") or None,
            max_chars=max(1, _env_int("OTTO_TTS_MAX_CHARS", 400)),
            trim_silence=os.environ.get("OTTO_TTS_TRIM_SILENCE", "1").strip().lower()
            in {"1", "true", "yes", "on"},
            pause_ms=max(0, _env_int("OTTO_TTS_PAUSE_MS", 280)),
            lead_keep_ms=max(0, _env_int("OTTO_TTS_LEAD_KEEP_MS", 12)),
            tail_keep_ms=max(0, _env_int("OTTO_TTS_TAIL_KEEP_MS", 25)),
            tail_hold_ms=max(0, _env_int("OTTO_TTS_TAIL_HOLD_MS", 150)),
        )

    @property
    def mode(self) -> str:
        """Which generation API the loaded weights expose."""
        if self.mock:
            return "mock"
        if self.backend == "chatterbox":
            return "chatterbox"
        name = self.model.lower()
        if "voicedesign" in name:
            return "design"
        if name.endswith("-base"):
            return "clone"
        return "custom"


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        return min(high, max(low, float(os.environ.get(name, default))))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass(slots=True)
class SpeechRequest:
    text: str
    voice: str
    instruct: str | None
    language: str
    speed: float


# ---------------------------------------------------------------------------
# backends (blocking: always called from a worker thread)
# ---------------------------------------------------------------------------
class MockBackend:
    """Fake 24 kHz generator (OTTO_TTS_MOCK=1) so the UI runs without a GPU."""

    warmup_s = 0.0

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def load(self) -> None:
        log.warning("mock backend: no model loaded, emitting a synthetic tone")

    def warmup(self) -> None:
        self.warmup_s = 0.0

    def stream(self, req: SpeechRequest) -> Iterator[np.ndarray]:
        seconds = min(6.0, max(0.4, len(req.text) / 14.0))
        total = int(seconds * SAMPLE_RATE)
        block = SAMPLE_RATE // 4
        for start in range(0, total, block):
            idx = np.arange(start, min(total, start + block), dtype=np.float32)
            warble = np.sin(2 * np.pi * 220.0 * idx / SAMPLE_RATE)
            warble *= np.sin(2 * np.pi * 3.0 * idx / SAMPLE_RATE)
            yield (0.12 * warble).astype(np.float32)


class QwenBackend:
    """faster-qwen3-tts backend; everything runs on ``cfg.device``."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model: Any = None
        self.warmup_s = 0.0
        self._rate_warned = False

    def load(self) -> None:
        if self.cfg.mode == "clone" and not (self.cfg.ref_audio and self.cfg.ref_text):
            raise ValueError(
                "clone mode needs OTTO_TTS_REF_AUDIO and OTTO_TTS_REF_TEXT (a -Base model clones a reference voice)"
            )
        compat.apply()
        from faster_qwen3_tts import FasterQwen3TTS

        log.info("loading %s on %s", self.cfg.model, self.cfg.device)
        self.model = FasterQwen3TTS.from_pretrained(self.cfg.model, device=self.cfg.device)
        self.warmup()

    def warmup(self) -> None:
        started = time.perf_counter()
        self.model.warmup()
        self.warmup_s = time.perf_counter() - started
        log.info("cuda graphs captured in %.2fs", self.warmup_s)

    def stream(self, req: SpeechRequest) -> Iterator[np.ndarray]:
        if self.model is None:
            raise RuntimeError("model not loaded")
        kwargs: dict[str, Any] = {
            "text": req.text,
            "language": req.language,
            "instruct": req.instruct,
            "chunk_size": self.cfg.chunk_size,
        }
        mode = self.cfg.mode
        if mode == "clone":
            generator = self.model.generate_voice_clone_streaming(
                ref_audio=self.cfg.ref_audio, ref_text=self.cfg.ref_text or "", **kwargs
            )
        elif mode == "design":
            generator = self.model.generate_voice_design_streaming(**kwargs)
        else:
            generator = self.model.generate_custom_voice_streaming(speaker=req.voice, **kwargs)
        for audio, rate, _timing in generator:
            if rate != SAMPLE_RATE and not self._rate_warned:
                self._rate_warned = True
                log.warning("model produced %s Hz audio, header says %s", rate, SAMPLE_RATE)
            yield np.asarray(audio, dtype=np.float32).reshape(-1)


def _time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Pitch-preserving WSOLA stretch; rate > 1 slows the speech down.

    Chatterbox's own pace knob barely moves a voice clone and pushing it high
    stiffens the delivery, so clips are stretched in the time domain instead:
    overlapping windows are re-aligned by correlation and cross-faded, which
    slows the speech while leaving pitch and timbre alone.
    """
    if abs(rate - 1.0) < 1e-3 or audio.size < 4096:
        return audio
    window = 1024  # ~43 ms at 24 kHz
    hop_in = window // 2
    hop_out = max(1, int(round(hop_in * rate)))
    search = 200
    shape = np.hanning(window).astype(np.float32)
    if audio.size < window + search + 1:
        return audio

    total = int(audio.size * rate) + 2 * window
    out = np.zeros(total, dtype=np.float32)
    norm = np.zeros(total, dtype=np.float32)
    overlap = window - hop_out
    views = np.lib.stride_tricks.sliding_window_view(audio, window)

    src = dst = 0
    while src + search < views.shape[0] and dst + window < total:
        if dst == 0:
            offset = 0
        else:
            reference = out[dst : dst + overlap]
            lo = max(0, src - search)
            hi = min(views.shape[0] - 1, src + search)
            if hi <= lo:
                break
            errors = np.einsum("ij,ij->i", views[lo : hi + 1, :overlap] - reference,
                               views[lo : hi + 1, :overlap] - reference)
            offset = lo + int(np.argmin(errors))
        out[dst : dst + window] += audio[offset : offset + window] * shape
        norm[dst : dst + window] += shape
        dst += hop_out
        src += hop_in
    out = out[: dst + window]
    return out / np.maximum(norm[: out.size], 1e-6)


class _NoWatermark:
    """Stand-in that passes audio through untouched."""

    def apply_watermark(self, wav, sample_rate=None, **_):  # noqa: ANN001
        return wav


class ChatterboxBackend:
    """ResembleAI Chatterbox — steadier prosody than the small Qwen model.

    One clip per request (no mid-generation streaming): time-to-first-audio is
    the synthesis of the first, deliberately short chunk, and a cancelled
    request finishes the in-flight clip — always a few seconds at most — before
    the lock frees. Speed maps onto ``cfg_weight``, Chatterbox's own pace
    control, so the voice speeds up without the pitch shift a resample gives;
    ``handles_speed`` tells the server not to resample on top.

    Voices are zero-shot clones: drop a reference clip into ``voices/<name>.wav``
    and it appears in ``/voices``; an unknown name falls back to the built-in
    default voice.
    """

    handles_speed = True

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model: Any = None
        self.warmup_s = 0.0

    def load(self) -> None:
        from chatterbox.tts import ChatterboxTTS

        log.info("loading Chatterbox on %s", self.cfg.device)
        self.model = ChatterboxTTS.from_pretrained(device=self.cfg.device)
        if not self.cfg.watermark:
            # perth embeds an inaudible-by-design provenance mark; in practice
            # it is the faint "noise" people hear at the very start of a clip
            self.model.watermarker = _NoWatermark()
            log.info("chatterbox watermark disabled")

    def warmup(self) -> None:
        """Prime the CUDA kernels with one tiny clip (conds already loaded)."""
        started = time.perf_counter()
        wav = self.model.generate(
            "Ready.",
            exaggeration=self.cfg.exaggeration,
            cfg_weight=self.cfg.cfg_weight,
        )
        del wav
        self.warmup_s = time.perf_counter() - started
        log.info("chatterbox warmup clip in %.2fs", self.warmup_s)

    def reference_for(self, voice: str) -> str | None:
        """Reference clip for a voice name; unknown names land on default.wav.

        The client keeps whatever voice the user last picked in localStorage
        ("Ryan", say), so a missing name must degrade to the shipped default
        voice rather than the model's rushed built-in conditionals.
        """
        base = Path(self.cfg.voices_dir)
        wanted = (voice or "").strip()
        for name in dict.fromkeys(filter(None, (wanted, "default"))):
            key = name.lower()
            for suffix in (".wav", ".mp3", ".flac", ".ogg"):
                candidate = base / (name + suffix)
                if candidate.is_file():
                    return str(candidate)
            # the UI keeps its own capitalisation ("Neil") — match stems case-insensitively
            if base.is_dir():
                for path in sorted(base.iterdir()):
                    if path.stem.lower() == key and path.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg"):
                        return str(path)
        return None

    def stream(self, req: SpeechRequest) -> Iterator[np.ndarray]:
        if self.model is None:
            raise RuntimeError("model not loaded")
        speed = float(req.speed or 1.0)
        cfg_weight = float(self.cfg.cfg_weight)
        if abs(speed - 1.0) > 1e-3:
            # inverse mapping: faster speech = lower cfg weight, pitch intact
            cfg_weight = min(0.8, max(0.3, cfg_weight / speed))
        kwargs: dict[str, Any] = {
            "text": req.text,
            "exaggeration": float(self.cfg.exaggeration),
            "cfg_weight": cfg_weight,
        }
        reference = self.reference_for(req.voice)
        if reference:
            kwargs["audio_prompt_path"] = reference
        wav = self.model.generate(**kwargs)
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        audio = _time_stretch(audio, self.cfg.stretch)
        step = 8192
        for i in range(0, len(audio), step):
            yield audio[i : i + step]


# ---------------------------------------------------------------------------
# engine: load state + single-flight lock
# ---------------------------------------------------------------------------
class Engine:
    """Owns the backend, the single-flight lock and the load/warmup bookkeeping."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        if cfg.mock:
            self.backend: Any = MockBackend(cfg)
        elif cfg.backend == "chatterbox":
            self.backend = ChatterboxBackend(cfg)
        else:
            self.backend = QwenBackend(cfg)
        self.lock = asyncio.Lock()
        self.ready = False
        self.error: str | None = None
        self.loaded_in_s = 0.0
        self.warmup_s: float | None = None
        self.started_at = time.time()

    async def load(self) -> None:
        """Load the model on a background thread; failures leave the API alive but not ready.

        The thread is a daemon on purpose: SIGINT while a 4 GB checkpoint is
        still being read must not hold the process open.
        """
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        finished: asyncio.Future[None] = loop.create_future()
        threading.Thread(
            target=_run_load, args=(self, loop, finished), name="tts-load", daemon=True
        ).start()
        try:
            await finished
        except Exception as exc:  # noqa: BLE001 - a broken model must not kill the service
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("model load failed; serving /health with ready=false")
        else:
            self.ready = True
            self.warmup_s = self.backend.warmup_s
            log.info("model ready in %.1fs (%s mode, %s)", time.perf_counter() - started, self.cfg.mode, self.cfg.device)
        finally:
            self.loaded_in_s = time.perf_counter() - started

    async def run_locked(self, fn, *args) -> asyncio.Task:
        """Run blocking ``fn`` in a worker thread, holding the lock until it exits.

        The lock is released from the task's done callback rather than by the
        caller: a cancelled HTTP response only sets the stop flag, so the GPU
        job is always finished before the next request may start.
        """
        await self.lock.acquire()
        try:
            job = asyncio.create_task(asyncio.to_thread(fn, *args))
        except BaseException:
            self.lock.release()
            raise
        job.add_done_callback(lambda _job: self.lock.release())
        return job

    async def warmup(self) -> float:
        """Capture the CUDA graphs; idempotent, single-flight."""
        if self.warmup_s is not None:
            return self.warmup_s
        job = await self.run_locked(self._capture_warmup)
        return float(await job)

    def _capture_warmup(self) -> float:
        self.backend.warmup()
        self.warmup_s = self.backend.warmup_s
        return self.warmup_s


def _run_load(engine: Engine, loop: asyncio.AbstractEventLoop, finished: asyncio.Future) -> None:
    """Body of the daemon load thread started by ``Engine.load``."""
    try:
        engine.backend.load()
    except BaseException as exc:  # noqa: BLE001 - reported through /health
        _settle(loop, finished, exc)
    else:
        _settle(loop, finished, None)


def _settle(loop: asyncio.AbstractEventLoop, finished: asyncio.Future, exc: BaseException | None) -> None:
    def complete() -> None:
        if finished.done():  # cancelled at shutdown
            return
        if exc is None:
            finished.set_result(None)
        else:
            finished.set_exception(exc)

    try:
        loop.call_soon_threadsafe(complete)
    except RuntimeError:  # event loop already closed during shutdown
        pass


# ---------------------------------------------------------------------------
# audio plumbing (blocking helpers, called from worker threads)
# ---------------------------------------------------------------------------
_END = object()


def _render(backend, req: SpeechRequest) -> Iterator[np.ndarray]:
    if getattr(backend, "handles_speed", False):
        # the backend has its own, pitch-preserving pace control
        yield from backend.stream(req)
        return
    yield from _apply_speed(backend.stream(req), req.speed)


def _apply_speed(chunks: Iterator[np.ndarray], speed: float) -> Iterator[np.ndarray]:
    """Resample when ``speed != 1.0``.

    faster-qwen3-tts has no tempo control, so speed is applied with soxr as a
    tape-style resample: 2.0 speaks twice as fast an octave higher. Pitch-exact
    time stretching is not worth a second dependency for a chat assistant.
    """
    if not speed or abs(speed - 1.0) < 1e-3:
        yield from chunks
        return
    import soxr

    stream = soxr.ResampleStream(int(SAMPLE_RATE * speed), SAMPLE_RATE, 1, dtype="float32")
    for chunk in chunks:
        audio = stream.resample_chunk(chunk)
        if audio.size:
            yield audio
    tail = stream.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
    if tail.size:
        yield tail


def _breath(pause_ms: int) -> bytes:
    """Fixed silence appended to every segment.

    Each synthesis call paces itself slightly differently, so whatever the
    model leaves at the end is trimmed and replaced with one identical breath —
    sentence transitions sound deliberate instead of random.
    """
    return b"\x00" * max(0, int(pause_ms * SAMPLE_RATE / 1000)) * 2


def _pcm16(chunk: np.ndarray) -> bytes:
    return (np.clip(chunk, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _post(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, item: Any) -> None:
    try:
        loop.call_soon_threadsafe(queue.put_nowait, item)
    except RuntimeError:  # event loop already closed during shutdown
        pass


def _pump(loop, queue, stop: threading.Event, backend, req: SpeechRequest) -> None:
    """Stream one utterance into ``queue`` as PCM16, stopping between chunks."""
    buffered = bytearray()
    try:
        chunks = _render(backend, req)
        try:
            for chunk in chunks:
                if stop.is_set() or SHUTDOWN.is_set():
                    break
                buffered += _pcm16(chunk)
                if len(buffered) >= PCM_CHUNK_BYTES:
                    _post(loop, queue, bytes(buffered))
                    buffered.clear()
            else:
                if buffered and not stop.is_set():
                    _post(loop, queue, bytes(buffered))
        finally:
            chunks.close()
    except BaseException as exc:  # noqa: BLE001 - surfaced to the response generator
        _post(loop, queue, exc)
    finally:
        _post(loop, queue, _END)


def _collect(backend, req: SpeechRequest) -> bytes:
    pcm = bytearray()
    chunks = _render(backend, req)
    try:
        for chunk in chunks:
            pcm += _pcm16(chunk)
    finally:
        chunks.close()
    return bytes(pcm)


def _wav_bytes(pcm: bytes) -> bytes:
    import soundfile as sf

    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# request validation
# ---------------------------------------------------------------------------
class SpeakBody(BaseModel):
    text: str = ""
    voice: str | None = None
    instruct: str | None = None
    language: str = "English"
    speed: float = 1.0


def _prepare_request(
    cfg: Config,
    text: str | None,
    voice: str | None,
    instruct: str | None,
    language: str | None,
    speed: float,
) -> SpeechRequest:
    clean = " ".join((text or "").split())
    if not clean:
        raise HTTPException(status_code=400, detail="text is required")
    if len(clean) > cfg.max_chars:
        clipped = clean[: cfg.max_chars]
        clean = clipped.rsplit(" ", 1)[0] if " " in clipped else clipped
        log.info("text clipped to %d chars (OTTO_TTS_MAX_CHARS=%d)", len(clean), cfg.max_chars)
    return SpeechRequest(
        text=clean,
        voice=_resolve_voice(cfg.mode, voice),
        instruct=(instruct or "").strip() or cfg.instruct or None,
        language=_resolve_language(language),
        speed=_resolve_speed(speed),
    )


def _resolve_voice(mode: str, voice: str | None) -> str:
    if mode == "clone":
        return CLONE_VOICE_ID
    if mode == "design":
        return DESIGN_VOICE_ID
    if mode == "chatterbox":
        return (voice or "").strip() or "default"
    wanted = (voice or "").strip().lower()
    if not wanted:
        return DEFAULT_VOICE
    for preset in PRESET_VOICES:
        if preset["id"].lower() == wanted:
            return preset["id"]
    raise HTTPException(status_code=400, detail=f"unknown voice {voice!r}; see /voices")


def _resolve_language(language: str | None) -> str:
    wanted = (language or "English").strip().lower()
    for known in SUPPORTED_LANGUAGES:
        if known.lower() == wanted:
            return known
    raise HTTPException(status_code=400, detail=f"unsupported language {language!r}")


def _resolve_speed(speed: float | None) -> float:
    try:
        value = float(speed if speed is not None else 1.0)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(value) or value <= 0:
        return 1.0
    return min(2.0, max(0.5, value))


def _voice_table(cfg: Config) -> tuple[str, list[dict[str, str]]]:
    if cfg.mode == "clone":
        return CLONE_VOICE_ID, [{
            "id": CLONE_VOICE_ID,
            "description": f"Cloned from {Path(cfg.ref_audio or 'the reference clip').name}",
            "language": "English",
            "kind": "clone",
        }]
    if cfg.mode == "design":
        return DESIGN_VOICE_ID, [{
            "id": DESIGN_VOICE_ID,
            "description": cfg.instruct,
            "language": "English",
            "kind": "design",
        }]
    if cfg.mode == "chatterbox":
        base = Path(cfg.voices_dir)
        default_ref = base / "default.wav"
        default_entry = {
            "id": "default",
            "description": f"cloned from {default_ref.name}" if default_ref.is_file() else "Chatterbox built-in voice",
            "language": "English",
            "kind": "clone" if default_ref.is_file() else "chatterbox",
        }
        table = [default_entry]
        if base.is_dir():
            for path in sorted(base.iterdir()):
                if path.stem.lower() == "default":
                    continue  # already listed via the default entry above
                if path.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg"):
                    table.append({
                        "id": path.stem,
                        "description": f"cloned from {path.name}",
                        "language": "English",
                        "kind": "clone",
                    })
        return "default", table
    return DEFAULT_VOICE, [dict(preset) for preset in PRESET_VOICES]


def _require_ready(engine: Engine) -> None:
    if not engine.ready:
        raise HTTPException(status_code=503, detail=engine.error or "model is loading")


def _gpu_name() -> str:
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch is optional in mock mode
        return "unavailable"
    try:
        if not torch.cuda.is_available():
            return "cpu"
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return "unknown"


class SilenceTrimmer:
    """Trim the dead air Qwen3-TTS puts around every segment.

    Each synthesis request starts and ends with a little silence. Concatenating
    those segments gives exactly the choppy, gappy delivery people notice when a
    voice assistant "switches sentences". Leading silence is dropped (keeping a
    short pad so the first phoneme is not clipped) and a small tail is held back
    so the trailing silence can be cut when the segment ends.
    """

    def __init__(self, cfg: Config) -> None:
        self.enabled = cfg.trim_silence
        self.pad_lead = int(cfg.lead_keep_ms * SAMPLE_RATE / 1000) * 2
        self.pad_tail = int(cfg.tail_keep_ms * SAMPLE_RATE / 1000) * 2
        self.hold = int(cfg.tail_hold_ms * SAMPLE_RATE / 1000) * 2
        self.fade = max(0, int(cfg.fade_in_ms * SAMPLE_RATE / 1000))
        self.threshold = int(0.006 * 32767)
        self.started = False
        self.held = bytearray()
        self.dropped_lead = 0
        self.dropped_tail = 0

    # ------------------------------------------------------------------ API
    def feed(self, data: bytes) -> bytes:
        if not self.enabled or not data:
            return data
        if not self.started:
            data, self.dropped_lead = self._trim_lead(data)
            if not data:
                return b""
            data = self._fade_in(data)
            self.started = True
        self.held += data
        if len(self.held) <= self.hold:
            return b""
        cut = len(self.held) - self.hold
        out = bytes(self.held[:cut])
        self.held = self.held[cut:]
        return out

    def flush(self) -> bytes:
        data = bytes(self.held)
        self.held = bytearray()
        if not self.enabled or not data:
            return data
        samples = memoryview(data).cast("h")
        last = len(samples) - 1
        while last >= 0 and abs(samples[last]) <= self.threshold:
            last -= 1
        if last < 0:
            self.dropped_tail = len(data)
            return b""
        end = min(len(samples), last + 1 + self.pad_tail // 2)
        self.dropped_tail = (len(samples) - end) * 2
        return bytes(data[: end * 2])

    # ------------------------------------------------------------- internals
    def _fade_in(self, data: bytes) -> bytes:
        """Linear ramp over the first samples — kills clicks and onset hiss."""
        if not self.fade or len(data) < 4:
            return data
        samples = bytearray(data)
        count = min(self.fade, len(samples) // 2)
        for i in range(count):
            gain = (i + 1) / count
            value = int(int.from_bytes(samples[i * 2 : i * 2 + 2], "little", signed=True) * gain)
            samples[i * 2 : i * 2 + 2] = value.to_bytes(2, "little", signed=True)
        return bytes(samples)

    def _trim_lead(self, data: bytes) -> tuple[bytes, int]:
        samples = memoryview(data).cast("h")
        for index, sample in enumerate(samples):
            if abs(sample) > self.threshold:
                start = max(0, index * 2 - self.pad_lead)
                return bytes(data[start:]), start
        return b"", len(data)


# ---------------------------------------------------------------------------
# streaming response
# ---------------------------------------------------------------------------
async def _speak_pcm(request: Request, engine: Engine, req: SpeechRequest) -> AsyncIterator[bytes]:
    """Yield PCM16LE chunks as the GPU produces them; abort on client disconnect."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    stop = threading.Event()
    started = time.perf_counter()
    job = await engine.run_locked(_pump, loop, queue, stop, engine.backend, req)
    pending = asyncio.create_task(queue.get())
    trimmer = SilenceTrimmer(engine.cfg)
    first_at: float | None = None
    finished_at: float | None = None
    samples = 0
    cancelled = False
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=DISCONNECT_POLL_S)
            if not done:
                if await request.is_disconnected():
                    cancelled = True
                    break
                continue
            item = pending.result()
            pending = asyncio.create_task(queue.get())
            if item is _END:
                finished_at = time.perf_counter()
                tail = trimmer.flush()
                if tail:
                    samples += len(tail) // 2
                    yield tail
                breath = _breath(engine.cfg.pause_ms)
                samples += len(breath) // 2
                yield breath
                break
            if isinstance(item, BaseException):
                log.error("generation failed after %d samples of audio: %r", samples, item)
                break
            outgoing = trimmer.feed(item)
            if not outgoing:
                continue
            if first_at is None:
                first_at = time.perf_counter()
            samples += len(outgoing) // 2
            yield outgoing
    except (asyncio.CancelledError, GeneratorExit):
        cancelled = True
        raise
    finally:
        stop.set()
        pending.cancel()
        end = finished_at or time.perf_counter()
        audio_ms = samples * 1000.0 / SAMPLE_RATE
        elapsed_ms = (end - started) * 1000.0
        log.info(
            "speak voice=%s chars=%d ttfa_ms=%.0f total_ms=%.0f audio_ms=%.0f rtf=%.2f cancelled=%s",
            req.voice,
            len(req.text),
            ((first_at or end) - started) * 1000.0,
            elapsed_ms,
            audio_ms,
            elapsed_ms / audio_ms if audio_ms else 0.0,
            cancelled,
        )
        if cancelled and not job.done():
            log.info("speak: client gone, stopping generation at the next chunk")


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = Engine(Config.from_env())
    app.state.engine = engine
    cfg = engine.cfg
    log.info(
        "otto tts: model=%s mode=%s device=%s mock=%s chunk_size=%d max_chars=%d",
        cfg.model, cfg.mode, cfg.device, cfg.mock, cfg.chunk_size, cfg.max_chars,
    )
    if cfg.ref_audio and cfg.mode != "clone":
        log.warning("OTTO_TTS_REF_AUDIO is set but %s is not a -Base model; ignoring it", cfg.model)
    loader = asyncio.create_task(engine.load(), name="tts-load")
    app.state.loader = loader
    try:
        yield
    finally:
        SHUTDOWN.set()
        if not loader.done():
            log.info("shutting down mid-load; abandoning the model thread")
            loader.cancel()
            try:
                await loader
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


app = FastAPI(title="Otto local TTS", version="1.0", lifespan=lifespan)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    engine: Engine = request.app.state.engine
    cfg = engine.cfg
    _default, voices = _voice_table(cfg)
    return {
        "ok": True,
        "backend": cfg.backend,
        "model": cfg.model,
        "device": cfg.device,
        "gpu": _gpu_name(),
        "sample_rate": SAMPLE_RATE,
        "mode": cfg.mode,
        "ready": engine.ready,
        "voices": len(voices),
        "uptime_s": round(time.time() - engine.started_at, 1),
        "loaded_in_s": round(engine.loaded_in_s, 2),
        "error": engine.error,
    }


@app.get("/voices")
async def voices(request: Request) -> dict[str, Any]:
    default, table = _voice_table(request.app.state.engine.cfg)
    return {"default": default, "voices": table}


@app.post("/speak")
async def speak(body: SpeakBody, request: Request) -> Response:
    engine: Engine = request.app.state.engine
    req = _prepare_request(engine.cfg, body.text, body.voice, body.instruct, body.language, body.speed)
    _require_ready(engine)
    return StreamingResponse(
        _speak_pcm(request, engine, req),
        media_type="audio/L16",
        headers={"X-Sample-Rate": str(SAMPLE_RATE), "Cache-Control": "no-store"},
    )


@app.get("/sample")
async def sample(
    request: Request,
    voice: str = DEFAULT_VOICE,
    text: str | None = None,
    instruct: str | None = None,
    language: str = "English",
    speed: float = 1.0,
) -> Response:
    engine: Engine = request.app.state.engine
    req = _prepare_request(engine.cfg, text or DEFAULT_SAMPLE_TEXT, voice, instruct, language, speed)
    _require_ready(engine)
    started = time.perf_counter()
    job = await engine.run_locked(_collect, engine.backend, req)
    pcm: bytes = await job
    if not pcm:
        raise HTTPException(status_code=500, detail="model produced no audio")
    audio_ms = len(pcm) / 2 * 1000.0 / SAMPLE_RATE
    log.info(
        "sample voice=%s chars=%d total_ms=%.0f audio_ms=%.0f rtf=%.2f",
        req.voice, len(req.text), (time.perf_counter() - started) * 1000.0, audio_ms,
        (time.perf_counter() - started) * 1000.0 / audio_ms if audio_ms else 0.0,
    )
    return Response(content=_wav_bytes(pcm), media_type="audio/wav")


@app.post("/warmup")
async def warmup(request: Request) -> dict[str, Any]:
    engine: Engine = request.app.state.engine
    _require_ready(engine)
    warmup_s = await engine.warmup()
    return {"ok": True, "warmup_s": round(warmup_s, 2)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Otto local Qwen3-TTS streaming service")
    parser.add_argument("--host", default=os.environ.get("OTTO_TTS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=_env_int("OTTO_TTS_PORT", 8801))
    parser.add_argument("--log-level", default=os.environ.get("OTTO_LOG_LEVEL", "info"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    import uvicorn

    # uvicorn owns SIGINT/SIGTERM: it stops accepting, closes the lifespan
    # (which sets SHUTDOWN so worker threads stop) and exits cleanly.
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
