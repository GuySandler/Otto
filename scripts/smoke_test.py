#!/usr/bin/env python
"""Headless end-to-end smoke test against a running Otto server.

    ./.venv/bin/python scripts/smoke_test.py [--url http://127.0.0.1:8800]

Checks REST health/bootstrap, then drives the websocket with typed text:
a plain answer, a tool call, and a barge-in. Saves the received audio to
/tmp/otto_smoke.wav and prints latency numbers.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import struct
import sys
import wave
from dataclasses import dataclass
from typing import Any

import httpx
import websockets


@dataclass
class SmokeSession:
    cookie: str
    email: str
    password: str
    temporary: bool


def _cookie_header(http: httpx.AsyncClient) -> str:
    return "; ".join(f"{key}={value}" for key, value in http.cookies.items())


async def authenticate(http: httpx.AsyncClient) -> SmokeSession:
    """Sign in with OTTO_SMOKE_EMAIL/PASSWORD, or create a throwaway account."""
    email = os.environ.get("OTTO_SMOKE_EMAIL", "").strip()
    password = os.environ.get("OTTO_SMOKE_PASSWORD", "").strip()
    if email and password:
        response = await http.post("/api/auth/login", json={"email": email, "password": password})
        if response.status_code != 200:
            raise SystemExit(f"login failed: {response.status_code} {response.text[:200]}")
        return SmokeSession(_cookie_header(http), email, password, False)

    email = f"smoke-{secrets.token_hex(4)}@otto.local"
    password = secrets.token_urlsafe(12)
    response = await http.post("/api/auth/signup", json={"email": email, "password": password, "name": "Smoke Test"})
    if response.status_code != 200:
        raise SystemExit(f"signup failed: {response.status_code} {response.text[:200]}")
    return SmokeSession(_cookie_header(http), email, password, True)


async def drop_temporary_account(base: str, session: SmokeSession) -> None:
    if not session.temporary:
        return
    async with httpx.AsyncClient(base_url=base, timeout=30) as http:
        http.headers["Cookie"] = session.cookie
        await http.request("DELETE", "/api/auth/account", json={"password": session.password})
    print("removed the throwaway account")


class Client:
    """Single-reader websocket client: one task owns recv(), everything else consumes events."""

    def __init__(self, url: str, *, cookie: str = "") -> None:
        self.url = url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self.cookie = cookie
        self.events: list[dict[str, Any]] = []
        self.audio = bytearray()
        self.sample_rate = 24000
        self.ws: Any = None
        self.reader: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "Client":
        headers = {"Cookie": self.cookie} if self.cookie else {}
        self.ws = await websockets.connect(self.url, max_size=16 * 1024 * 1024, additional_headers=headers)
        self.reader = asyncio.create_task(self._read(), name="smoke-reader")
        await self.send({"type": "hello"})
        await self.wait_for("ready", timeout=15)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self.reader is not None:
            self.reader.cancel()
        if self.ws is not None:
            await self.ws.close()

    async def send(self, payload: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(payload))

    async def _read(self) -> None:
        try:
            async for message in self.ws:
                if isinstance(message, bytes):
                    magic, seq, rate, index = struct.unpack("<4sIII", message[:16])
                    if magic != b"AUD1":
                        raise AssertionError("bad audio frame magic")
                    self.sample_rate = rate
                    self.audio.extend(message[16:])
                else:
                    self.events.append(json.loads(message))
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            return

    async def wait_for(self, kind: str, timeout: float = 60.0) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            for index, event in enumerate(self.events):
                if event["type"] == kind:
                    return self.events.pop(index)
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError(f"timed out waiting for {kind}; saw {[e['type'] for e in self.events]}")
            await asyncio.sleep(0.02)

    async def settle(self, seconds: float = 0.4) -> None:
        await asyncio.sleep(seconds)

    def drain(self) -> list[dict[str, Any]]:
        out, self.events = self.events, []
        return out

    def last_state(self) -> str | None:
        for event in reversed(self.events):
            if event["type"] == "state":
                return event["state"]
        return None

    def save_wav(self, path: str) -> float:
        if not self.audio:
            return 0.0
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(bytes(self.audio))
        return len(self.audio) / 2 / self.sample_rate


async def run(base: str) -> int:
    failures: list[str] = []
    async with httpx.AsyncClient(base_url=base, timeout=30) as http:
        session = await authenticate(http)
        print(f"signed in as {session.email}")
        health = (await http.get("/api/health")).json()
        print("health:", json.dumps(health)[:300])
        if not health.get("lunaroute", {}).get("configured"):
            print("!! LUNAROUTE_API_KEY missing — aborting")
            return 2
        bootstrap = (await http.get("/api/bootstrap")).json()
        print("models:", [m["id"] for m in bootstrap["models"]][:6])
        print("connections:", {k: v.get("connected") for k, v in bootstrap["connections"].items()})
        print("tools:", [t["name"] for t in bootstrap["tools"] if t["configured"]])

    async with Client(base, cookie=session.cookie) as client:
        # 1) plain spoken answer
        client.audio.clear()
        await client.send({"type": "text", "text": "Say hello back to me in one short sentence."})
        message = await client.wait_for("assistant.message")
        print("\n[1] assistant:", message["text"])
        await client.settle(1.2)
        seconds = client.save_wav("/tmp/otto_smoke.wav")
        print(f"    audio: {seconds:.2f}s @ {client.sample_rate} Hz -> /tmp/otto_smoke.wav")
        metrics = [e for e in client.drain() if e["type"] == "metrics"]
        if metrics:
            print("    metrics:", metrics[0]["latency"])
        if seconds <= 0:
            failures.append("no audio came back (is the local TTS running?)")

        # 2) a tool call
        await client.send({"type": "text", "text": "What time is it right now? Use your tool."})
        try:
            call = await client.wait_for("tool.call", timeout=45)
            print("\n[2] tool call:", call["name"], call["args"])
            result = await client.wait_for("tool.result", timeout=45)
            print("    tool result ok:", result["ok"], "|", result["summary"][:120])
        except TimeoutError as exc:
            failures.append(f"tool call failed: {exc}")
        await client.wait_for("assistant.message", timeout=60)

        # 3) barge-in
        await client.send({"type": "text", "text": "Count slowly from one to twenty, out loud."})
        await asyncio.sleep(1.2)
        await client.send({"type": "barge_in"})
        try:
            cancel = await client.wait_for("cancel.audio", timeout=10)
            print("\n[3] barge-in flushed audio from seq", cancel["fromSeq"])
            await client.settle(0.6)
            state = client.last_state()
            print("    state after barge-in:", state)
            if state != "listening":
                failures.append(f"state after barge-in was {state!r}, expected 'listening'")
        except TimeoutError as exc:
            failures.append(f"barge-in failed: {exc}")

    await drop_temporary_account(base, session)

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("\nall smoke checks passed")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8800")
    args = parser.parse_args()
    try:
        code = asyncio.run(run(args.url))
    except Exception as exc:  # noqa: BLE001
        print(f"smoke test crashed: {type(exc).__name__}: {exc}")
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
