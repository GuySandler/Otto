"""Lunaroute speech-to-text (parakeet-tdt-0.6b-v3)."""
from __future__ import annotations

import io
import logging
import time
import wave
from dataclasses import dataclass

import httpx

log = logging.getLogger("otto.stt")

SAMPLE_RATE = 16_000


class STTError(RuntimeError):
    pass


@dataclass(slots=True)
class Transcript:
    text: str
    latency_ms: int
    duration_ms: int


def pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


class LunarouteSTT:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.model = settings.stt_model
        self.language = settings.cfg("stt", "language", default=None)

    @property
    def configured(self) -> bool:
        return bool(self.settings.lunaroute_api_key)

    async def transcribe(
        self,
        pcm: bytes,
        *,
        sample_rate: int = SAMPLE_RATE,
        language: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> Transcript:
        if not pcm:
            return Transcript(text="", latency_ms=0, duration_ms=0)
        if not self.configured:
            raise STTError("LUNAROUTE_API_KEY is not set")
        started = time.perf_counter()
        data = {"model": model or self.model, "response_format": "json"}
        lang = language or self.language
        if lang:
            data["language"] = lang
        headers = {"Authorization": f"Bearer {self.settings.lunaroute_api_key}"}
        files = {"file": ("utterance.wav", pcm_to_wav(pcm, sample_rate), "audio/wav")}
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(
                    f"{self.settings.lunaroute_base_url}/audio/transcriptions",
                    headers=headers,
                    data=data,
                    files=files,
                )
                resp.raise_for_status()
                payload = resp.json()
            except httpx.HTTPError as exc:
                raise STTError(f"transcription failed: {exc}") from exc
        text = (payload.get("text") or "").strip()
        duration_ms = int(len(pcm) / 2 / sample_rate * 1000)
        return Transcript(
            text=text,
            latency_ms=int((time.perf_counter() - started) * 1000),
            duration_ms=duration_ms,
        )
