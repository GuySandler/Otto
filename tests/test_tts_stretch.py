"""The WSOLA time-stretch: slower speech without a pitch shift."""
from __future__ import annotations

import numpy as np

from tts.server import _time_stretch


def _zero_cross_rate(x: np.ndarray, sr: int = 24000) -> float:
    crossings = int(np.count_nonzero(np.diff(np.signbit(x))))
    return crossings / 2.0 / (len(x) / sr)


def test_stretch_slows_the_audio():
    sr = 24000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out = _time_stretch(tone, 1.25)
    assert 2.3 < len(out) / sr < 2.7, f"expected ~25% longer, got {len(out)/sr:.2f}s"


def test_stretch_preserves_pitch():
    sr = 24000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out = _time_stretch(tone, 1.25)
    freq = _zero_cross_rate(out)
    assert abs(freq - 220) < 18, f"pitch drifted: {freq:.1f} Hz"


def test_noop_on_rate_one():
    x = np.random.default_rng(1).standard_normal(24000).astype(np.float32)
    out = _time_stretch(x, 1.0)
    assert out is x
