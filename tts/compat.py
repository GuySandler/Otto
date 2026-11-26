"""Runtime compatibility shims for the local Qwen3-TTS stack.

``qwen-tts-hf`` (the Transformers-5 build used by ``faster-qwen3-tts``) assigns
``self.upsample`` / ``self.decoder*`` in a ``MimiModel`` subclass, but
Transformers >= 5 exposes those names as read-only properties, so instantiation
dies with ``property ... object has no setter``. Shadowing the inherited
property with a plain class attribute restores the original behaviour.
"""
from __future__ import annotations

import inspect
import logging

log = logging.getLogger("otto.tts.compat")


def apply() -> list[str]:
    patched: list[str] = []
    try:
        import qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 as tok
    except Exception:  # noqa: BLE001 - optional dependency
        return patched

    for cls_name in ("Qwen3TTSTokenizerV2Encoder",):
        cls = getattr(tok, cls_name, None)
        if not isinstance(cls, type):
            continue
        for attr in ("upsample", "decoder_transformer", "decoder", "encoder_transformer", "encoder"):
            static = inspect.getattr_static(cls, attr, None)
            if isinstance(static, property):
                setattr(cls, attr, None)
                patched.append(f"{cls_name}.{attr}")
    if patched:
        log.info("compat shims applied: %s", ", ".join(patched))
    return patched
