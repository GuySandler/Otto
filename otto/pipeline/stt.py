"""Lunaroute speech-to-text (whisper-large-v3)."""
from __future__ import annotations

import asyncio
import io
import logging
import time
import wave
from dataclasses import dataclass

import httpx

log = logging.getLogger("otto.stt")

SAMPLE_RATE = 16_000

#: gateway/upstream hiccup codes worth one more try; anything else (401, 413,
#: 422 …) fails immediately — retrying cannot change the answer
_TRANSIENT = {429, 500, 502, 503, 504}


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
    def __init__(self, settings, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self.model = settings.stt_model
        self.language = settings.cfg("stt", "language", default=None)
        # ordered failover ladder: the primary first, then whatever the config
        # lists — one model being down on the gateway must not mute the user
        raw = settings.cfg("stt", "fallback_models", default=None) or []
        self.fallback_models = [str(m).strip() for m in raw if str(m).strip()]
        self._transport = transport  # test seam

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
        attempts: int = 3,
    ) -> Transcript:
        """Transcribe one utterance, surviving gateway hiccups two ways.

        A 502 is usually a single model's upstream being briefly unhealthy:
        first the same model is retried (short backoff), then the configured
        fallback models take over — losing a spoken sentence to a flaky
        gateway is unacceptable.
        """
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
        client_kwargs: dict = {"timeout": timeout}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        ladder: list[str] = []
        for name in [model or self.model, *self.fallback_models]:
            if name and name not in ladder:
                ladder.append(name)

        last_error: STTError | None = None
        async with httpx.AsyncClient(**client_kwargs) as client:
            for model_name in ladder:
                data = {"model": model_name, "response_format": "json"}
                if lang:
                    data["language"] = lang
                try:
                    payload = await self._post_once(client, headers, data, files, attempts)
                except STTError as exc:
                    last_error = exc
                    log.info("stt: %s unusable (%s) — moving on", model_name, str(exc)[:80])
                    continue
                return Transcript(
                    text=(payload.get("text") or "").strip(),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    duration_ms=int(len(pcm) / 2 / sample_rate * 1000),
                )
        assert last_error is not None
        raise last_error

    async def _post_once(
        self, client: httpx.AsyncClient, headers: dict, data: dict, files: dict, attempts: int
    ) -> dict:
        """One model, up to ``attempts`` tries on transient failures."""
        for attempt in range(max(1, attempts)):
            try:
                resp = await client.post(
                    f"{self.settings.lunaroute_base_url}/audio/transcriptions",
                    headers=headers,
                    data=data,
                    files=files,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in _TRANSIENT or attempt == attempts - 1:
                    raise STTError(f"transcription failed: {exc}") from exc
                log.info("stt: HTTP %s (attempt %d/%d) — retrying", status, attempt + 1, attempts)
            except httpx.HTTPError as exc:
                if attempt == attempts - 1:
                    raise STTError(f"transcription failed: {exc}") from exc
                log.info("stt: %s (attempt %d/%d) — retrying", type(exc).__name__, attempt + 1, attempts)
            await asyncio.sleep(0.6 * (attempt + 1))
        raise STTError("transcription failed: no response")
