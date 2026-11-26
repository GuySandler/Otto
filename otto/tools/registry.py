"""Tool discovery.

Any module in ``otto.tools`` (or an extra path from ``OTTO_TOOL_PLUGINS``) that
exposes ``TOOLS: list[Tool]`` is picked up at startup.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import pkgutil
import sys
from pathlib import Path
from typing import Any

from .base import Tool

log = logging.getLogger("otto.tools")

_SKIP = {"base", "registry", "__init__"}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ------------------------------------------------------------------ access
    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            log.warning("duplicate tool %s ignored", tool.name)
            return
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(
        self,
        *,
        only_available: bool = True,
        flags: dict[str, bool] | None = None,
        ctx: Any = None,
        exclude: set[str] | None = None,
        exclude_categories: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            t.to_schema()
            for t in self._tools.values()
            if (not only_available or t.is_available(ctx))
            and _flag_on(t, flags)
            and (not exclude or t.name not in exclude)
            and (not exclude_categories or t.category not in exclude_categories)
        ]

    def catalog(self, flags: dict[str, bool] | None = None, ctx: Any = None) -> list[dict[str, Any]]:
        out = []
        for tool in sorted(self._tools.values(), key=lambda t: (t.category, t.name)):
            entry = tool.to_catalog(ctx)
            entry["enabled"] = _flag_on(tool, flags)
            out.append(entry)
        return out

    def by_category(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for item in self.catalog():
            out.setdefault(item["category"], []).append(item)
        return out

    # ---------------------------------------------------------------- discovery
    @classmethod
    def discover(cls, plugin_paths: list[str] | None = None) -> "ToolRegistry":
        registry = cls()
        package = sys.modules[__package__]
        for module_info in pkgutil.iter_modules(package.__path__):
            if module_info.name in _SKIP:
                continue
            full = f"{__package__}.{module_info.name}"
            try:
                module = importlib.import_module(full)
            except Exception:  # noqa: BLE001 - one broken module must not break startup
                log.exception("failed to import tool module %s", full)
                continue
            registry._load_module(module)

        extra = plugin_paths if plugin_paths is not None else _plugin_paths()
        for path in extra:
            registry._load_file(Path(path))
        return registry

    def _load_module(self, module: Any) -> None:
        tools = getattr(module, "TOOLS", None)
        if not tools:
            return
        for tool in tools:
            if isinstance(tool, Tool):
                self.add(tool)
            else:
                log.warning("%s: TOOLS entry is not a Tool: %r", module.__name__, tool)

    def _load_file(self, path: Path) -> None:
        if not path.exists():
            log.warning("tool plugin not found: %s", path)
            return
        name = f"otto_tool_plugin_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:  # noqa: BLE001
            log.exception("failed to load tool plugin %s", path)
            return
        self._load_module(module)


def _flag_on(tool: Tool, flags: dict[str, bool] | None) -> bool:
    if tool.flag is None:
        return True
    return bool((flags or {}).get(tool.flag))


def _plugin_paths() -> list[str]:
    raw = os.environ.get("OTTO_TOOL_PLUGINS", "")
    return [p.strip() for p in raw.split(",") if p.strip()]
