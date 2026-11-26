"""Settings: .env secrets + config/app.yaml knobs."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader (no dependency, does not override real env vars)."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


@dataclass(slots=True)
class Settings:
    root: Path = ROOT
    # lunaroute
    lunaroute_api_key: str = ""
    lunaroute_base_url: str = "https://gw.lunaroute.com/v1"
    stt_model: str = "parakeet-tdt-0.6b-v3"
    llm_model: str = "deepseek-4.1-flash"
    # local tts
    tts_enabled: bool = True
    tts_url: str = "http://127.0.0.1:8801"
    tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    tts_gpu: int = 1
    # capability defaults (the UI toggles these per session)
    parallel_agents: bool = False
    background_model: bool = False
    #: MCP overrides (empty → value from config/app.yaml)
    mcp_enabled: bool | None = None
    mcp_mode: str = ""
    #: sqlite file (users, sessions, per-user secrets)
    db_path: str = ""
    # server
    host: str = "127.0.0.1"
    port: int = 8800
    log_level: str = "info"
    # integrations
    google_client_secrets: str = "secrets/google_client.json"
    google_client_id: str = ""
    google_client_secret: str = ""
    #: explicit base URL for links/OAuth redirects (empty → http://host:port)
    public_url_override: str = ""
    google_token_path: str = "secrets/google_token.json"
    gcal_calendar_id: str = "primary"
    slack_bot_token: str = ""
    slack_default_channel: str = ""
    # config tree (from app.yaml)
    app_cfg: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- helpers
    @property
    def data_dir(self) -> Path:
        path = self.root / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def database_path(self) -> Path:
        return self.path(self.db_path) if self.db_path else self.data_dir / "otto.db"

    @property
    def secrets_dir(self) -> Path:
        path = self.root / "secrets"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def web_dir(self) -> Path:
        return self.root / "web"

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    def path(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    @property
    def public_url(self) -> str:
        """Base URL the browser uses — must match the registered OAuth redirect URI."""
        explicit = str(self.public_url_override or "").strip().rstrip("/")
        return explicit or f"http://{self.host}:{self.port}"

    @property
    def google_client_secrets_path(self) -> Path:
        return self.path(self.google_client_secrets)

    @property
    def google_token_file(self) -> Path:
        return self.path(self.google_token_path)

    # ------------------------------------------------------------------- yaml
    def cfg(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.app_cfg
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def models(self) -> list[str]:
        models = self.cfg("lunaroute", "models", default=[]) or []
        out = [str(m) for m in models]
        if self.llm_model and self.llm_model not in out:
            out.insert(0, self.llm_model)
        return out


def load_settings(root: Path | None = None) -> Settings:
    root = Path(root or ROOT)
    _load_dotenv(root / ".env")
    cfg_path = root / "config" / "app.yaml"
    app_cfg: dict[str, Any] = {}
    if cfg_path.exists():
        app_cfg = yaml.safe_load(cfg_path.read_text()) or {}

    settings = Settings(
        root=root,
        lunaroute_api_key=_env("LUNAROUTE_API_KEY"),
        lunaroute_base_url=_env("LUNAROUTE_BASE_URL", app_cfg.get("lunaroute", {}).get("base_url", "https://gw.lunaroute.com/v1")).rstrip("/"),
        stt_model=_env("OTTO_STT_MODEL", app_cfg.get("stt", {}).get("model", "parakeet-tdt-0.6b-v3")),
        llm_model=_env("OTTO_LLM_MODEL", app_cfg.get("lunaroute", {}).get("default_model", "deepseek-4.1-flash")),
        tts_enabled=_env_bool("OTTO_TTS_ENABLED", bool(app_cfg.get("tts", {}).get("enabled", True))),
        tts_url=_env("OTTO_TTS_URL", app_cfg.get("tts", {}).get("url", "http://127.0.0.1:8801")).rstrip("/"),
        tts_model=_env("OTTO_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        tts_gpu=_env_int("OTTO_TTS_GPU", 1),
        mcp_enabled=_env_bool("OTTO_MCP_ENABLED", True) if _env("OTTO_MCP_ENABLED") else None,
        mcp_mode=_env("OTTO_MCP_MODE"),
        db_path=_env("OTTO_DB_PATH"),
        parallel_agents=_env_bool("OTTO_PARALLEL_AGENTS", False),
        background_model=_env_bool("OTTO_BACKGROUND_MODEL", False),
        host=_env("OTTO_HOST", app_cfg.get("app", {}).get("host", "127.0.0.1")),
        port=_env_int("OTTO_PORT", int(app_cfg.get("app", {}).get("port", 8800))),
        log_level=_env("OTTO_LOG_LEVEL", "info"),
        google_client_secrets=_env("GOOGLE_CLIENT_SECRETS", "secrets/google_client.json"),
        google_client_id=_env("GOOGLE_CLIENT_ID"),
        google_client_secret=_env("GOOGLE_CLIENT_SECRET"),
        public_url_override=_env("OTTO_PUBLIC_URL"),
        google_token_path=_env("GOOGLE_TOKEN_PATH", "secrets/google_token.json"),
        gcal_calendar_id=_env("OTTO_GCAL_CALENDAR_ID", "primary"),
        slack_bot_token=_env("SLACK_BOT_TOKEN"),
        slack_default_channel=_env("SLACK_DEFAULT_CHANNEL"),
        app_cfg=app_cfg,
    )
    return settings


_current: Settings | None = None


def set_current(settings: Settings) -> Settings:
    """Remember the process-wide settings object (used by tool modules)."""
    global _current
    _current = settings
    return settings


def get_current() -> Settings:
    global _current
    if _current is None:
        _current = load_settings()
    return _current
