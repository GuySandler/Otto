"""Persistence for UI-selectable defaults (model, voice, personality)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .protocol import Config


class ConfigStore:
    def __init__(self, path: Path, *, defaults: Config) -> None:
        self.path = path
        self.defaults = defaults

    def load(self) -> Config:
        config = Config(**self.defaults.to_dict())
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
            except ValueError:
                data = {}
            config.update(data)
        return config

    def save(self, config: Config) -> None:
        self.path.write_text(json.dumps(config.to_dict(), indent=2))
