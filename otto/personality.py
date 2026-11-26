"""Personality loading.

A personality is a YAML file in ``config/personalities``. User edits made from
the UI are written to ``data/personalities/<id>.yaml`` and shadow the built-ins,
so the shipped personas stay pristine.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")
    return slug or "personality"


DEFAULT_FILLERS = ["One sec.", "Let me check.", "Mm-hm."]


@dataclass(slots=True)
class Personality:
    id: str
    name: str
    system_prompt: str
    tagline: str = ""
    description: str = ""
    voice: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    speech: dict[str, Any] = field(default_factory=dict)
    source: str = "builtin"

    # -------------------------------------------------------------- rendering
    @property
    def fillers(self) -> list[str]:
        fillers = self.speech.get("fillers") or DEFAULT_FILLERS
        return [str(f) for f in fillers if str(f).strip()]

    @property
    def thinking_fillers(self) -> list[str]:
        fillers = self.speech.get("thinking_fillers") or ["Mm.", "Okay.", "Well."]
        return [str(f) for f in fillers if str(f).strip()]

    @property
    def tool_filler(self) -> str:
        return str(self.speech.get("tool_filler") or self.fillers[0])

    def build_system_prompt(self, *, global_rules: str = "", extra: str = "") -> str:
        parts = [self.system_prompt.strip()]
        style = str(self.speech.get("style") or "").strip()
        if style:
            parts.append(f"Style notes:\n{style}")
        if global_rules.strip():
            parts.append(global_rules.strip())
        if extra.strip():
            parts.append(extra.strip())
        return "\n\n".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tagline": self.tagline,
            "description": self.description,
            "voice": copy.deepcopy(self.voice),
            "llm": copy.deepcopy(self.llm),
            "speech": copy.deepcopy(self.speech),
            "system_prompt": self.system_prompt,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: str = "custom") -> "Personality":
        pid = _slug(str(data.get("id") or data.get("name") or "personality"))
        return cls(
            id=pid,
            name=str(data.get("name") or pid.title()),
            system_prompt=str(data.get("system_prompt") or "").strip(),
            tagline=str(data.get("tagline") or ""),
            description=str(data.get("description") or ""),
            voice=dict(data.get("voice") or {}),
            llm=dict(data.get("llm") or {}),
            speech=dict(data.get("speech") or {}),
            source=source,
        )


class PersonalityStore:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.builtin_dir = settings.config_dir / "personalities"
        self.user_dir = settings.data_dir / "personalities"
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Personality] = {}

    # ------------------------------------------------------------------ paths
    def _files(self, directory: Path) -> dict[str, Path]:
        if not directory.exists():
            return {}
        return {p.stem: p for p in sorted(directory.glob("*.y*ml"))}

    def paths(self) -> dict[str, Path]:
        merged = self._files(self.builtin_dir)
        merged.update(self._files(self.user_dir))
        return merged

    # ------------------------------------------------------------------- load
    def load(self, pid: str, *, use_cache: bool = True) -> Personality:
        pid = _slug(pid)
        if use_cache and pid in self._cache:
            return self._cache[pid]
        paths = self.paths()
        if pid not in paths:
            raise KeyError(pid)
        path = paths[pid]
        data = yaml.safe_load(path.read_text()) or {}
        data.setdefault("id", pid)
        source = "user" if path.parent == self.user_dir else "builtin"
        personality = Personality.from_dict(data, source=source)
        personality.id = pid
        self._cache[pid] = personality
        return personality

    def default(self, preferred: str = "otto") -> Personality:
        try:
            return self.load(preferred)
        except KeyError:
            paths = self.paths()
            if not paths:
                return Personality(id="otto", name="Otto", system_prompt="You are Otto, a helpful voice assistant.")
            return self.load(next(iter(paths)))

    def all(self) -> list[Personality]:
        out: list[Personality] = []
        for pid in self.paths():
            try:
                out.append(self.load(pid))
            except Exception:  # a broken user file must not kill the app
                continue
        return out

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "id": p.id,
                "name": p.name,
                "tagline": p.tagline,
                "voice": p.voice.get("voice", ""),
                "model": p.llm.get("model", ""),
                "source": p.source,
            }
            for p in self.all()
        ]

    # ------------------------------------------------------------------ write
    def save(self, pid: str, data: dict[str, Any]) -> Personality:
        pid = _slug(pid)
        fields = {"name", "tagline", "description", "voice", "llm", "speech", "system_prompt"}
        # merge over what the personality resolves to today, so a partial update
        # from the UI (name + prompt) keeps its voice, fillers and model
        try:
            base = self.load(pid, use_cache=False).to_dict()
        except KeyError:
            base = {}
        payload = {k: base[k] for k in fields if base.get(k) is not None}
        payload.update({k: v for k, v in data.items() if k in fields})
        payload["id"] = pid
        path = self.user_dir / f"{pid}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
        self._cache.pop(pid, None)
        return self.load(pid, use_cache=False)
