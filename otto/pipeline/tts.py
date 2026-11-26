"""Client for the local Qwen3-TTS service (tts/server.py, pinned to GPU1)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

log = logging.getLogger("otto.tts")

DEFAULT_SAMPLE_RATE = 24_000


class TTSError(RuntimeError):
    pass


@dataclass(slots=True)
class TTSChunk:
    pcm: bytes
    sample_rate: int


class LocalTTS:
    """Streams raw PCM16LE mono from the local model server.

    Cancelling the async generator (barge-in) aborts the HTTP request, which the
    TTS server observes as a disconnect and uses to stop GPU generation.
    """

    def __init__(self, settings) -> None:
        self.settings = settings
        self.base_url = settings.tts_url
        self.timeout = float(settings.cfg("tts", "request_timeout_s", default=60))
        self._healthy: bool | None = None
        self._voices: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return bool(self.settings.tts_enabled)

    # ------------------------------------------------------------------ health
    async def health(self, *, refresh: bool = False) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "detail": "disabled in config"}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/health")
                resp.raise_for_status()
                data = resp.json()
            self._healthy = bool(data.get("ok", True))
            return data
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            self._healthy = False
            return {"ok": False, "detail": str(exc)}

    async def voices(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        if self._voices and not refresh:
            return self._voices
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/voices")
                resp.raise_for_status()
                payload = resp.json()
            self._voices = list(payload.get("voices") or [])
        except Exception as exc:  # noqa: BLE001
            log.info("tts voices unavailable: %s", exc)
            self._voices = []
        return self._voices

    # ------------------------------------------------------------------- speak
    async def stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        instruct: str | None = None,
        language: str | None = None,
        speed: float = 1.0,
    ) -> AsyncIterator[TTSChunk]:
        if not text.strip():
            return
        if not self.enabled:
            raise TTSError("tts disabled")
        body = {
            "text": text,
            "voice": voice or None,
            "instruct": instruct or None,
            "language": language or "English",
            "speed": speed,
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5.0)) as client:
            try:
                async with client.stream("POST", f"{self.base_url}/speak", json=body) as resp:
                    if resp.status_code >= 400:
                        detail = (await resp.aread()).decode(errors="replace")[:300]
                        raise TTSError(f"tts {resp.status_code}: {detail}")
                    sample_rate = int(resp.headers.get("x-sample-rate") or DEFAULT_SAMPLE_RATE)
                    async for chunk in resp.aiter_raw(chunk_size=4096):
                        if chunk:
                            yield TTSChunk(pcm=chunk, sample_rate=sample_rate)
            except httpx.HTTPError as exc:
                raise TTSError(f"tts request failed: {exc}") from exc
