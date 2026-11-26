#!/usr/bin/env python
"""Voice sanity check: synth → transcribe → compare.

    ./.venv/bin/python scripts/voice_check.py --voice Aiden

Asks the local Qwen3-TTS service for a few sentences, saves them under
data/voice-check/, measures duration / RMS / time-to-first-byte, and (when
LUNAROUTE_API_KEY is set) transcribes them back with parakeet so you can see
whether the voice is actually intelligible rather than just loud.

This is the fastest way to audition speakers or tweak the `instruct` prompt
without clicking through the UI.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import io
import sys
import time
import wave
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from otto.settings import load_settings  # noqa: E402

SENTENCES = [
    "Hey, good to see you. I'm Otto, your voice assistant.",
    "You've got two emails from Sarah; the newest one is about the lease renewal.",
    "There's a meeting with the design team tomorrow at ten, and another one on Friday afternoon.",
    "Do you want me to send that message, or should I hold off?",
]


def wav_bytes_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        return wav.readframes(wav.getnframes()), wav.getframerate()


def resample(pcm: bytes, src: int, dst: int) -> bytes:
    import numpy as np

    audio = np.frombuffer(pcm, dtype="<i2").astype("float32")
    if src == dst or audio.size == 0:
        return pcm
    n_out = int(round(audio.size * dst / src))
    x_out = np.linspace(0, audio.size - 1, n_out)
    out = np.interp(x_out, np.arange(audio.size), audio)
    return out.astype("<i2").tobytes()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8801")
    parser.add_argument("--voice", default="Aiden")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--out", default="data/voice-check")
    args = parser.parse_args()

    settings = load_settings()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stt = None
    if settings.lunaroute_api_key:
        from otto.pipeline.stt import LunarouteSTT

        stt = LunarouteSTT(settings)
    else:
        print("(no LUNAROUTE_API_KEY — skipping the transcription half of the check)")

    ok = True
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            health = (await client.get(f"{args.url}/health")).json()
        except Exception as exc:  # noqa: BLE001
            print(f"tts is not reachable at {args.url}: {exc}")
            return 2
        print("tts:", {k: health.get(k) for k in ("model", "mode", "device", "ready", "sample_rate")})

        for index, sentence in enumerate(SENTENCES):
            started = time.perf_counter()
            first = None
            chunks: list[bytes] = []
            async with client.stream(
                "POST",
                f"{args.url}/speak",
                json={"text": sentence, "voice": args.voice, "instruct": args.instruct or None},
            ) as resp:
                resp.raise_for_status()
                rate = int(resp.headers.get("x-sample-rate", 24000))
                async for chunk in resp.aiter_raw(512):
                    if first is None:
                        first = time.perf_counter() - started
                    chunks.append(chunk)
            total = time.perf_counter() - started
            pcm = b"".join(chunks)
            samples = len(pcm) // 2
            dur = samples / rate
            import numpy as np

            audio = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
            rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
            path = out_dir / f"{args.voice.lower()}-{index}.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(pcm)
            print(
                f"[{index}] {dur:5.2f}s audio | ttfb {first * 1000 if first else -1:6.0f} ms | "
                f"total {total:5.2f}s | rtf {total / dur if dur else 0:.2f} | rms {rms:.3f} → {path}"
            )
            if rms < 0.01:
                print("    !! nearly silent output")
                ok = False
            if stt is not None:
                pcm16 = resample(pcm, rate, 16000)
                try:
                    result = await stt.transcribe(pcm16)
                except Exception as exc:  # noqa: BLE001
                    print(f"    transcription failed: {exc}")
                    continue
                ratio = difflib.SequenceMatcher(None, sentence.lower(), result.text.lower()).ratio()
                flag = "ok " if ratio > 0.75 else "?? "
                print(f"    {flag}heard: {result.text!r}  (match {ratio:.2f})")
                if ratio <= 0.75:
                    ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
