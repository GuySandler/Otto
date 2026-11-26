"""Wire protocol primitives."""
from __future__ import annotations

import json

from otto.protocol import (
    AUDIO_MAGIC,
    Config,
    ToolCall,
    ToolCalls,
    audio_frame,
    decode_client,
    parse_audio_frame,
)


def test_audio_frame_roundtrip():
    payload = b"\x01\x02\x03\x04"
    frame = audio_frame(seq=7, sample_rate=24000, index=3, payload=payload)
    assert frame[:4] == AUDIO_MAGIC
    seq, rate, index, pcm = parse_audio_frame(frame)
    assert (seq, rate, index, pcm) == (7, 24000, 3, payload)


def test_parse_rejects_garbage():
    assert parse_audio_frame(b"") is None
    assert parse_audio_frame(b"NOPE" + b"\x00" * 20) is None


def test_tool_calls_accumulate_deltas():
    acc = ToolCalls()
    acc.add({"index": 0, "id": "call_1", "function": {"name": "gmail_send"}})
    acc.add({"index": 0, "function": {"arguments": '{"to": ["a@b.c"],'}})
    acc.add({"index": 0, "function": {"arguments": '"subject": "hi"}'}})
    acc.add({"index": 1, "id": "call_2", "function": {"name": "get_current_time", "arguments": "{}"}})
    calls = acc.finish()
    assert [c.name for c in calls] == ["gmail_send", "get_current_time"]
    assert calls[0].args() == {"to": ["a@b.c"], "subject": "hi"}


def test_tool_call_args_survive_bad_json():
    call = ToolCall(id="x", name="n", arguments="{not json")
    assert call.args() == {}


def test_config_update_ignores_wrong_types():
    config = Config()
    config.update({"model": "glm-5.3", "speed": 1.2, "max_tokens": "nope", "bogus": 1})
    assert config.model == "glm-5.3"
    assert config.speed == 1.2
    assert isinstance(config.max_tokens, int)


def test_decode_client_handles_junk():
    assert decode_client("null") == {"type": "invalid"}
    assert decode_client("{oops") == {"type": "invalid"}
    assert decode_client(json.dumps({"type": "ping"})) == {"type": "ping"}
