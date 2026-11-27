"""The STT client must ride out transient gateway failures."""
from __future__ import annotations

import httpx
import pytest

from otto.pipeline.stt import STTError, LunarouteSTT

PCM = b"\x00\x01" * 1600  # 0.1 s of silence, enough to be non-empty


class StubSettings:
    stt_model = "whisper-large-v3"
    lunaroute_api_key = "test-key"
    lunaroute_base_url = "https://gw.test"

    def cfg(self, *args, **kwargs):
        if "fallback_models" in args:
            return ["parakeet-tdt-0.6b-v3"]
        return None


def stt_with(handler, *, fallbacks=True) -> tuple[LunarouteSTT, list]:
    calls: list[bytes] = []

    def wrapped(request):
        calls.append(request.read())
        return handler(request)

    settings = StubSettings()
    if not fallbacks:
        settings.cfg = lambda *a, **k: None
    return LunarouteSTT(settings, transport=httpx.MockTransport(wrapped)), calls


async def test_a_transient_502_is_retried():
    n = []

    def handler(request):
        n.append(1)
        if len(n) == 1:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, json={"text": "hello there"})

    client, calls = stt_with(handler)
    transcript = await client.transcribe(PCM)
    assert transcript.text == "hello there"
    assert len(n) == 2, "the 502 should have been retried once, then succeeded"


async def test_permanent_errors_fail_fast():
    def handler(request):
        return httpx.Response(401, text="bad key")

    client, calls = stt_with(handler, fallbacks=False)
    with pytest.raises(STTError):
        await client.transcribe(PCM)
    assert len(calls) == 1, "a 401 must not be retried on the same model"


async def test_gives_up_after_three_attempts():
    def handler(request):
        return httpx.Response(503, text="down")

    client, calls = stt_with(handler, fallbacks=False)
    with pytest.raises(STTError):
        await client.transcribe(PCM)
    assert len(calls) == 3


async def test_falls_over_to_the_backup_model():
    """whisper down (502) -> the configured parakeet backup answers."""
    bodies: list[bytes] = []

    def handler(request):
        bodies.append(request.read())
        if b"whisper-large-v3" in bodies[-1]:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, json={"text": "hello from parakeet"})

    client, calls = stt_with(handler)
    transcript = await client.transcribe(PCM)
    assert transcript.text == "hello from parakeet"
    assert any(b"parakeet-tdt-0.6b-v3" in body for body in bodies), "the fallback model was never requested"


async def test_all_models_down_raises():
    def handler(request):
        return httpx.Response(503, text="down")

    client, calls = stt_with(handler)
    with pytest.raises(STTError):
        await client.transcribe(PCM)
