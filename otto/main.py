"""FastAPI app: auth, websocket voice loop, REST surface and the static UI."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .actions import ActionStore
from .config_store import ConfigStore
from .db import Database
from .google_oauth import GoogleOAuth, GoogleOAuthError
from .integrations import Integrations
from .mcp import MCPManager
from .personality import PersonalityStore
from .pipeline.llm import LunarouteLLM
from .pipeline.stt import LunarouteSTT
from .pipeline.tts import LocalTTS
from .protocol import Config, decode_client
from .session import Session
from .settings import Settings, load_settings, set_current
from .tools.base import ToolContext
from .tools.registry import ToolRegistry
from .users import SESSION_COOKIE, AuthError, User, UserStore

log = logging.getLogger("otto")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

#: paths reachable without a session
PUBLIC_PATHS = {"/login", "/favicon.ico", "/api/health"}
PUBLIC_PREFIXES = (
    "/static/",
    "/api/auth/",
    "/oauth/google/callback",
)


class AppState:
    """Everything a session needs, built once at startup."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = Database(settings.database_path)
        self.users = UserStore(self.db)
        self.actions = ActionStore(self.db)
        self.google_oauth = GoogleOAuth(settings, self.users)
        self.llm = LunarouteLLM(settings)
        self.stt = LunarouteSTT(settings)
        self.tts = LocalTTS(settings)
        self.personalities = PersonalityStore(settings)
        self.registry = ToolRegistry.discover()
        self.config_store = ConfigStore(settings.data_dir / "config.json", defaults=self._defaults())
        self.config = self.config_store.load()
        self.started_at = time.time()

    def _defaults(self) -> Config:
        personality = self.personalities.default()
        defaults = Config(
            model=self.settings.llm_model,
            personality=personality.id,
            temperature=float(self.settings.cfg("lunaroute", "temperature", default=0.7)),
            max_tokens=int(self.settings.cfg("lunaroute", "max_tokens", default=500)),
            parallel_agents=self.settings.parallel_agents,
            background_model=self.settings.background_model,
            stt_model=self.settings.stt_model,
        )
        defaults.voice = str(personality.voice.get("voice") or "Aiden")
        defaults.instruct = str(personality.voice.get("instruct") or "")
        defaults.language = str(personality.voice.get("language") or "English")
        defaults.speed = float(personality.voice.get("speed") or 1.0)
        return defaults

    # ------------------------------------------------------------- per-user bits
    def integrations_for(self, user: User | None) -> Integrations:
        return Integrations(self.settings, users=self.users, user_id=user.id if user else None)

    def tool_context(self, user: User | None) -> ToolContext:
        return ToolContext(settings=self.settings, integrations=self.integrations_for(user))

    def strict_mcp(self) -> bool:
        """True when only MCP may touch Workspace (no legacy REST tools at all)."""
        cfg = self.settings.cfg("mcp", default={}) or {}
        enabled = bool(cfg.get("enabled", True)) if getattr(self.settings, "mcp_enabled", None) is None else bool(self.settings.mcp_enabled)
        mode = str(getattr(self.settings, "mcp_mode", "") or cfg.get("mode", "mcp") or "mcp").lower()
        return enabled and mode in ("mcp", "auto") and not bool(cfg.get("allow_rest_fallback", False))

    def visible_tools(self, user: User | None) -> list[dict[str, Any]]:
        """Registry tools minus the ones strict MCP mode replaces."""
        ctx = self.tool_context(user)
        catalog = self.registry.catalog(ctx=ctx)
        if self.strict_mcp():
            # gmail/calendar = the legacy REST tools; Google's MCP tools replace them
            catalog = [entry for entry in catalog if entry["category"] not in ("gmail", "calendar")]
        return catalog

    def mcp_manager(self, user: User | None) -> MCPManager:
        integrations = self.integrations_for(user)

        async def token() -> str:
            return await integrations.google.access_token()

        return MCPManager(self.settings, token_provider=token)

    async def mcp_snapshot(self, user: User | None) -> dict[str, Any]:
        """Discover MCP tools for a user (page loads / the "check again" button)."""
        manager = self.mcp_manager(user)
        timeout = float(self.settings.cfg("mcp", "discover_timeout_s", default=8))
        try:
            await asyncio.wait_for(manager.discover(), timeout=timeout)
        except asyncio.TimeoutError:
            log.info("mcp discovery timed out")
        except Exception:  # noqa: BLE001
            log.exception("mcp discovery failed")
        status = manager.status()          # includes the tool *count*
        status["catalog"] = manager.catalog()  # and the tool descriptors
        await manager.aclose()
        return status

    def connections(self, user: User | None) -> dict[str, Any]:
        google = self.google_oauth.status(user.id if user else None)
        slack: dict[str, Any] = {"configured": False, "connected": False, "team": None, "detail": ""}
        if user is not None:
            try:
                slack = self.integrations_for(user).slack.status()
            except Exception as exc:  # noqa: BLE001
                slack = {"configured": False, "connected": False, "team": None, "detail": str(exc)}
        return {"google": google, "slack": slack}

    async def aclose(self) -> None:
        self.db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = set_current(load_settings())
    state = AppState(settings)
    app.state.otto = state
    state.users.purge_expired()
    health = await state.tts.health()
    log.info(
        "Otto %s ready — llm=%s stt=%s tools=%d tts=%s users=%d",
        __version__,
        "configured" if state.llm.configured else "NO API KEY",
        state.config.stt_model or settings.stt_model,
        len(state.registry),
        "up" if health.get("ok") else f"down ({health.get('detail', 'unknown')})",
        state.users.count(),
    )
    if not state.llm.configured:
        log.warning("LUNAROUTE_API_KEY is empty — add it to .env")
    if not state.google_oauth.configured:
        log.info("Google OAuth client not configured — see docs/AUTH.md to enable Connections")
    yield
    await state.aclose()


app = FastAPI(title="Otto", version=__version__, lifespan=lifespan)


def state_of(request: Request | WebSocket) -> AppState:
    return request.app.state.otto


# ---------------------------------------------------------------------------
# auth plumbing
# ---------------------------------------------------------------------------
def user_from(request: Request | WebSocket) -> User | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    return state_of(request).users.user_for_token(token)


def require_user(request: Request) -> User:
    user = getattr(request.state, "user", None) or user_from(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not signed in")
    return user


def set_session_cookie(response: Response, token: str, *, secure: bool) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=60 * 60 * 24 * 30,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    path = request.url.path
    user = user_from(request)
    request.state.user = user
    if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
        return await call_next(request)
    if user is None:
        # pages redirect to the login screen, API calls answer 401 so clients can react
        is_page = not path.startswith(("/api/", "/oauth/", "/ws"))
        if is_page or "text/html" in request.headers.get("accept", ""):
            return RedirectResponse("/login", status_code=302)
        return JSONResponse({"detail": "Not signed in"}, status_code=401)
    # cheap CSRF guard: a mutating request from another origin is refused
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        origin = request.headers.get("origin")
        if origin:
            host = request.headers.get("host", "")
            if origin.split("://")[-1].rstrip("/") != host:
                return JSONResponse({"detail": "Cross-origin request refused"}, status_code=403)
    return await call_next(request)


# ---------------------------------------------------------------------------
# auth routes
# ---------------------------------------------------------------------------
@app.post("/api/auth/signup")
async def signup(request: Request) -> Response:
    state = state_of(request)
    payload = await request.json()
    try:
        user = state.users.create_user(
            str(payload.get("email") or ""),
            str(payload.get("password") or ""),
            str(payload.get("name") or ""),
        )
    except AuthError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)
    token = state.users.new_session(user.id, user_agent=request.headers.get("user-agent", ""))
    response = JSONResponse({"user": user.to_dict()})
    set_session_cookie(response, token, secure=request.url.scheme == "https")
    return response


@app.post("/api/auth/login")
async def login(request: Request) -> Response:
    state = state_of(request)
    payload = await request.json()
    try:
        user, token = state.users.login(
            str(payload.get("email") or ""),
            str(payload.get("password") or ""),
            user_agent=request.headers.get("user-agent", ""),
        )
    except AuthError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)
    response = JSONResponse({"user": user.to_dict()})
    set_session_cookie(response, token, secure=request.url.scheme == "https")
    return response


@app.post("/api/auth/logout")
async def logout(request: Request) -> Response:
    state = state_of(request)
    state.users.drop_session(request.cookies.get(SESSION_COOKIE))
    response = JSONResponse({"ok": True})
    clear_session_cookie(response)
    return response


@app.delete("/api/auth/account")
async def delete_account(request: Request) -> Response:
    state = state_of(request)
    user = require_user(request)
    payload = await request.json()
    try:
        state.users.delete_user(user.id, str(payload.get("password") or ""))
    except AuthError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)
    response = JSONResponse({"ok": True})
    clear_session_cookie(response)
    return response


@app.get("/api/auth/me")
async def me(request: Request) -> Response:
    user = getattr(request.state, "user", None)
    if user is None:
        return JSONResponse({"detail": "Not signed in"}, status_code=401)
    return JSONResponse({"user": {**user.to_dict(), "initials": user.initials}})


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health(request: Request) -> dict[str, Any]:
    state = state_of(request)
    user = getattr(request.state, "user", None)
    tts = await state.tts.health()
    connections = state.connections(user) if user is not None else {}
    return {
        "ok": True,
        "version": __version__,
        "uptime_s": int(time.time() - state.started_at),
        "lunaroute": {"configured": state.llm.configured, "model": state.config.model},
        "stt": {"model": state.config.stt_model or state.settings.stt_model},
        "tts": tts,
        "tools": len(state.registry),
        "auth": {"signedIn": user is not None, "users": state.users.count()},
        **connections,
    }


@app.get("/api/bootstrap")
async def bootstrap(request: Request) -> dict[str, Any]:
    state = state_of(request)
    user = require_user(request)
    models: list[dict[str, Any]] = []
    stt_models: list[dict[str, Any]] = []
    if state.llm.configured:
        try:
            models = await asyncio.wait_for(state.llm.models(), timeout=8.0)
            stt_models = await asyncio.wait_for(state.llm.transcription_models(), timeout=8.0)
        except Exception as exc:  # noqa: BLE001
            log.info("model list unavailable: %s", exc)
    if not models:
        models = [{"id": m, "label": m, "default": m == state.config.model} for m in state.settings.models()]
    if not stt_models:
        stt_models = [{"id": state.settings.stt_model, "label": state.settings.stt_model}]
    voices = await state.tts.voices()
    mcp_status = await state.mcp_snapshot(user) if user is not None else {"enabled": False, "servers": [], "tools": []}
    tools_catalog = state.visible_tools(user) + [
        {**entry, "source": "mcp"} for entry in mcp_status.pop("catalog", [])
    ]
    mcp_status["strict"] = state.strict_mcp()
    return {
        "app": {"name": state.settings.cfg("app", "name", default="Otto"), "version": __version__},
        "user": {**user.to_dict(), "initials": user.initials},
        "models": models,
        "sttModels": stt_models,
        "personalities": state.personalities.summary(),
        "voices": voices,
        "tools": tools_catalog,
        "mcp": mcp_status,
        "actions": {"open": state.actions.count(user.id)},
        "config": state.config.to_dict(),
        "connections": state.connections(user),
        "capabilities": {
            "parallel_agents": True,
            "background_model": {
                "available": state.llm.has_background_variant(state.config.model),
                "id": state.llm.background_variant(state.config.model),
            },
            "jobs": True,
        },
        "status": await health(request),
    }


@app.post("/api/config")
async def set_config(request: Request) -> dict[str, Any]:
    state = state_of(request)
    require_user(request)
    patch = await request.json()
    if not isinstance(patch, dict):
        raise HTTPException(400, "expected a JSON object")
    state.config.update(patch)
    state.config_store.save(state.config)
    return {"ok": True, "config": state.config.to_dict()}


@app.get("/api/tools")
async def tools(request: Request) -> dict[str, Any]:
    state = state_of(request)
    user = require_user(request)
    mcp_status = await state.mcp_snapshot(user)
    catalog = state.visible_tools(user) + [{**entry, "source": "mcp"} for entry in mcp_status.pop("catalog", [])]
    categories: dict[str, list[dict[str, Any]]] = {}
    for entry in catalog:
        categories.setdefault(entry["category"], []).append(entry)
    mcp_status["strict"] = state.strict_mcp()
    return {"tools": catalog, "categories": categories, "mcp": mcp_status}


@app.post("/api/mcp/selftest")
async def mcp_selftest(request: Request) -> dict[str, Any]:
    """One real read call per MCP server, with this account's Google token."""
    state = state_of(request)
    user = require_user(request)
    manager = state.mcp_manager(user)
    ctx = state.tool_context(user)
    try:
        return await asyncio.wait_for(manager.self_test(ctx), timeout=60)
    except asyncio.TimeoutError:
        return {"ok": False, "reason": "timeout", "detail": "the MCP servers did not answer in 45s"}
    finally:
        await manager.aclose()


@app.post("/api/mcp/probe")
async def mcp_probe(request: Request) -> dict[str, Any]:
    """Re-run MCP discovery now (the Connections card's "check again")."""
    state = state_of(request)
    user = require_user(request)
    return await state.mcp_snapshot(user)


@app.get("/api/personalities")
async def personalities(request: Request) -> dict[str, Any]:
    state = state_of(request)
    require_user(request)
    return {"personalities": state.personalities.summary()}


@app.get("/api/personalities/{pid}")
async def get_personality(request: Request, pid: str) -> dict[str, Any]:
    state = state_of(request)
    require_user(request)
    try:
        return state.personalities.load(pid).to_dict()
    except KeyError:
        raise HTTPException(404, "unknown personality") from None


@app.put("/api/personalities/{pid}")
async def put_personality(request: Request, pid: str) -> dict[str, Any]:
    state = state_of(request)
    require_user(request)
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(400, "expected a JSON object")
    return state.personalities.save(pid, payload).to_dict()


# ---------------------------------------------------------------------------
# connections (per-user credentials)
# ---------------------------------------------------------------------------
@app.get("/api/connections")
async def get_connections(request: Request) -> dict[str, Any]:
    state = state_of(request)
    user = require_user(request)
    return state.connections(user)


@app.delete("/api/connections/google")
async def disconnect_google(request: Request) -> dict[str, Any]:
    state = state_of(request)
    user = require_user(request)
    state.users.delete_google(user.id)
    return state.connections(user)


@app.post("/api/connections/slack")
async def connect_slack(request: Request) -> dict[str, Any]:
    state = state_of(request)
    user = require_user(request)
    payload = await request.json()
    token = str(payload.get("token") or "").strip()
    if not token:
        raise HTTPException(400, "a Slack bot token is required")
    if not token.startswith(("xoxb-", "xoxp-", "xoxa-", "xoxr-")):
        raise HTTPException(400, "that doesn't look like a Slack token (expected xoxb-…)")
    state.users.set_secret(user.id, "slack_bot_token", token)
    # verify it with one cheap call so a bad token does not look connected
    client = state.integrations_for(user).slack
    try:
        await client.channels(limit=1)
    except Exception as exc:  # noqa: BLE001
        state.users.delete_secret(user.id, "slack_bot_token")
        raise HTTPException(400, f"Slack rejected that token: {exc}") from exc
    finally:
        await client.aclose()
    return state.connections(user)


@app.delete("/api/connections/slack")
async def disconnect_slack(request: Request) -> dict[str, Any]:
    state = state_of(request)
    user = require_user(request)
    state.users.delete_secret(user.id, "slack_bot_token")
    return state.connections(user)


# ---------------------------------------------------------------------------
# actions (things the user still has to finish)
# ---------------------------------------------------------------------------
@app.get("/api/actions")
async def list_actions(request: Request, status: str = "open") -> dict[str, Any]:
    state = state_of(request)
    user = require_user(request)
    wanted = status if status in {"open", "done", "dismissed", "all"} else "open"
    items = state.actions.list(user.id, status=wanted)
    return {
        "actions": [action.to_dict() for action in items],
        "counts": {
            "open": state.actions.count(user.id),
            "done": state.actions.count(user.id, status="done"),
            "dismissed": state.actions.count(user.id, status="dismissed"),
        },
    }


@app.post("/api/actions/{action_id}")
async def update_action(request: Request, action_id: str) -> dict[str, Any]:
    state = state_of(request)
    user = require_user(request)
    payload = await request.json()
    status = str(payload.get("status") or "")
    if status not in {"open", "done", "dismissed"}:
        raise HTTPException(400, "status must be open, done or dismissed")
    action = state.actions.set_status(user.id, action_id, status)
    if action is None:
        raise HTTPException(404, "unknown action")
    return {"action": action.to_dict(), "counts": {"open": state.actions.count(user.id)}}


# ---------------------------------------------------------------------------
# Google OAuth
# ---------------------------------------------------------------------------
@app.get("/oauth/google/start")
async def google_start(request: Request) -> Response:
    state = state_of(request)
    user = require_user(request)
    if not state.google_oauth.configured:
        raise HTTPException(400, "Google OAuth client is not configured (see docs/AUTH.md)")
    url = state.google_oauth.authorization_url(state.google_oauth.new_state(user.id))
    return RedirectResponse(url, status_code=302)


@app.get("/oauth/google/callback")
async def google_callback(request: Request, code: str = "", state: str = "", error: str = "") -> Response:
    app_state = state_of(request)
    user = user_from(request)
    if user is None:
        return RedirectResponse("/login", status_code=302)
    if error:
        app_state.google_oauth.set_error(user.id, error, f"Google returned '{error}' from the consent screen")
        return RedirectResponse(f"/?error=google_{error}", status_code=302)
    if not code:
        return RedirectResponse("/?error=google_no_code", status_code=302)
    if not app_state.google_oauth.check_state(user.id, state):
        return RedirectResponse("/?error=google_bad_state", status_code=302)
    try:
        token_json, email, scopes = await asyncio.to_thread(app_state.google_oauth.exchange, code)
    except GoogleOAuthError as exc:
        log.warning("google oauth exchange failed (%s): %s", getattr(exc, "reason", "unknown"), exc)
        reason = getattr(exc, "reason", "unknown")
        app_state.google_oauth.set_error(user.id, reason, str(exc))
        return RedirectResponse(f"/?error=google_exchange_failed&reason={reason}", status_code=302)
    app_state.users.save_google(user.id, token_json, email=email, scopes=scopes)
    app_state.google_oauth.clear_error(user.id)
    log.info("google connected for %s (%s)", user.email, email or "unknown account")
    return RedirectResponse("/?connected=google", status_code=302)


@app.get("/api/voice-sample")
async def voice_sample(request: Request, voice: str = "Aiden", text: str | None = None) -> Response:
    state = state_of(request)
    require_user(request)
    sample = text or "Hi, I'm Otto. This is how I sound when we talk."
    try:
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(f"{state.tts.base_url}/sample", params={"voice": voice, "text": sample})
            resp.raise_for_status()
            return Response(content=resp.content, media_type="audio/wav")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"tts unavailable: {exc}") from exc


@app.post("/api/warmup")
async def warmup(request: Request) -> dict[str, Any]:
    state = state_of(request)
    require_user(request)
    try:
        import httpx

        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(f"{state.tts.base_url}/warmup")
            resp.raise_for_status()
            return {"ok": True, **resp.json()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"tts unavailable: {exc}") from exc


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    state = state_of(ws)
    user = user_from(ws)
    if user is None:
        await ws.close(code=4401)
        return

    send_lock = asyncio.Lock()

    async def send_json(payload: dict[str, Any]) -> None:
        async with send_lock:
            try:
                await ws.send_text(json.dumps(payload, default=str))
            except (RuntimeError, WebSocketDisconnect):
                pass

    async def send_bytes(data: bytes) -> None:
        async with send_lock:
            try:
                await ws.send_bytes(data)
            except (RuntimeError, WebSocketDisconnect):
                pass

    integrations = state.integrations_for(user)
    session = Session(
        session_id=uuid.uuid4().hex[:12],
        settings=state.settings,
        llm=state.llm,
        stt=state.stt,
        tts=state.tts,
        registry=state.registry,
        personalities=state.personalities,
        integrations=integrations,
        config=Config(**state.config.to_dict()),
        send_json=send_json,
        send_bytes=send_bytes,
        user_id=user.id,
        actions=state.actions,
    )
    control: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def control_worker() -> None:
        while True:
            msg = await control.get()
            try:
                if msg is None:
                    return
                await session.handle_message(msg)
            except Exception:  # noqa: BLE001
                log.exception("control message failed")
            finally:
                control.task_done()

    worker = asyncio.create_task(control_worker(), name="otto-control")
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data:
                await session.on_audio(data)
                continue
            text = message.get("text")
            if text:
                payload = decode_client(text)
                kind = payload.get("type")
                # latency-critical control messages must not wait behind the
                # transcribe/turn work queued in the control worker
                if kind in ("utterance.start", "barge_in"):
                    try:
                        await session.handle_message(payload)
                    except Exception:  # noqa: BLE001
                        log.exception("inline control failed")
                    continue
                await control.put(payload)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception("websocket error")
    finally:
        await control.put(None)
        try:
            await asyncio.wait_for(worker, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            worker.cancel()
        await session.close()
        await integrations.aclose()


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------
def _web_file(name: str) -> Path:
    return WEB_DIR / name


@app.get("/login")
async def login_page(request: Request) -> Any:
    if user_from(request) is not None:
        return RedirectResponse("/", status_code=302)
    page = _web_file("login.html")
    if not page.exists():
        return JSONResponse({"detail": "web/login.html missing"}, status_code=500)
    return FileResponse(page)


@app.get("/dashboard")
async def dashboard(request: Request) -> Any:
    page = _web_file("dashboard.html")
    if not page.exists():
        return JSONResponse({"detail": "web/dashboard.html missing"}, status_code=500)
    return FileResponse(page)


@app.get("/")
async def index(request: Request) -> Any:
    page = _web_file("index.html")
    if not page.exists():
        return JSONResponse({"detail": "web/ not built"}, status_code=500)
    return FileResponse(page)


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def main() -> None:
    import uvicorn

    settings = load_settings()
    uvicorn.run(
        "otto.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
