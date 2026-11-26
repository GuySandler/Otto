"""Per-user MCP tool discovery and invocation.

One manager per conversation. It connects to the configured MCP servers *as the
signed-in user* (their Google access token), lists the tools once, hands them to
the harness as ordinary :class:`otto.tools.base.Tool` objects, and routes calls
back over MCP.

Failure is always explained rather than swallowed: "your OAuth client isn't
marked as an AI agent" is a message a human can act on, and the UI shows it.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..tools.base import Tool, ToolContext, ToolResult
from .client import MCPClient, MCPError, MCPToolInfo, explain_tool_error
from .policy import confirmation_for, policy_for

log = logging.getLogger("otto.mcp")


@dataclass(slots=True)
class ServerState:
    name: str
    url: str
    ok: bool = False
    reason: str = ""
    detail: str = ""
    tools: list[MCPToolInfo] = field(default_factory=list)
    checked_at: float = 0.0

    def to_dict(self, *, include_tools: bool = False) -> dict[str, Any]:
        out = {
            "name": self.name,
            "url": self.url,
            "ok": self.ok,
            "reason": self.reason,
            "detail": self.detail,
            "tools": len(self.tools),
        }
        if include_tools:
            out["toolNames"] = [tool.name for tool in self.tools]
        return out


class MCPManager:
    """Discovers and calls MCP tools for one user/session."""

    def __init__(
        self,
        settings,
        *,
        token_provider: Callable[[], Awaitable[str]] | None = None,
        transports: dict[str, Any] | None = None,
        fallback_lookup: Callable[[str], Any] | None = None,
    ) -> None:
        cfg = settings.cfg("mcp", default={}) or {}
        self.settings = settings
        override_enabled = getattr(settings, "mcp_enabled", None)
        self.enabled = bool(cfg.get("enabled", True)) if override_enabled is None else bool(override_enabled)
        self.mode = str(getattr(settings, "mcp_mode", "") or cfg.get("mode", "mcp") or "mcp").lower()
        self.allow_rest_fallback = bool(cfg.get("allow_rest_fallback", True))
        self.timeout = float(cfg.get("timeout_s", 30))
        self.cache_ttl = float(cfg.get("tool_cache_s", 600))
        self.prune_schemas = bool(cfg.get("prune_schemas", True))
        self.max_description = int(cfg.get("max_schema_description", 110))
        self.servers_cfg: dict[str, dict[str, Any]] = dict(cfg.get("servers") or {})
        self.token_provider = token_provider
        self.transports = transports or {}
        #: resolves the REST twin of an MCP tool (same Otto name) for per-call fallback
        self.fallback_lookup = fallback_lookup
        self._clients: dict[str, MCPClient] = {}
        self._states: dict[str, ServerState] = {}
        self._tools: dict[str, Tool] = {}
        self._info: dict[str, MCPToolInfo] = {}
        self._checked = 0.0
        self._token_ready = False
        self._token_error = ""
        self._lock = asyncio.Lock()
        self._notices: list[str] = []
        self._last_error: dict[str, Any] | None = None

    # ------------------------------------------------------------------ config
    @property
    def uses_mcp(self) -> bool:
        return self.enabled and self.mode in ("mcp", "auto") and bool(self.servers_cfg)

    @property
    def authoritative(self) -> bool:
        """True when MCP replaces the REST tools outright (no fallback)."""
        return self.uses_mcp and (self.mode == "mcp" and not self.allow_rest_fallback)

    def server_names(self) -> list[str]:
        return list(self.servers_cfg.keys())

    def _client(self, name: str) -> MCPClient:
        if name not in self._clients:
            cfg = self.servers_cfg.get(name) or {}
            self._clients[name] = MCPClient(
                str(cfg.get("url") or ""),
                name=name,
                token_provider=self.token_provider,
                timeout=float(cfg.get("timeout_s") or self.timeout),
                transport=self.transports.get(name),
            )
        return self._clients[name]

    # --------------------------------------------------------------- discovery
    async def discover(self, *, force: bool = False) -> list[Tool]:
        if not self.uses_mcp:
            return []
        async with self._lock:
            fresh = (time.time() - self._checked) < self.cache_ttl
            if self._tools and fresh and not force:
                return list(self._tools.values())

            # Google's MCP servers answer tools/list anonymously and only require
            # the bearer token for the calls themselves, so discovery runs either
            # way and we gate *use* on the token instead.
            self._token_ready = False
            if self.token_provider is not None:
                try:
                    await self.token_provider()
                    self._token_ready = True
                except Exception as exc:  # noqa: BLE001
                    self._token_error = str(exc)

            tools: dict[str, Tool] = {}
            for name in self.server_names():
                client = self._client(name)
                state = ServerState(name=name, url=client.url)
                try:
                    discovered = self._curate(name, await client.list_tools())
                    state.ok = True
                    state.tools = discovered
                    for info in discovered:
                        tool = self._build_tool(name, info)
                        if tool is None:
                            continue
                        if tool.name in tools:
                            log.info("mcp: %s overrides an earlier %s tool", tool.name, name)
                        tools[tool.name] = tool
                        self._info[f"{name}.{info.name}"] = info
                    log.info("mcp: %s exposed %d tool(s)", name, len(discovered))
                except MCPError as exc:
                    state.reason = exc.reason
                    state.detail = str(exc)
                    log.warning("mcp: %s unavailable (%s): %s", name, exc.reason, exc)
                    self._notices.append(str(exc))
                except Exception as exc:  # noqa: BLE001
                    state.reason = "error"
                    state.detail = f"{type(exc).__name__}: {exc}"
                    log.exception("mcp: %s discovery crashed", name)
                finally:
                    state.checked_at = time.time()
                    self._states[name] = state

            self._tools = tools
            self._checked = time.time()
            return list(tools.values())

    def _curate(self, server: str, tools: list[MCPToolInfo]) -> list[MCPToolInfo]:
        """Keep only the tools this server's config asks for.

        Google's servers publish 20+ tools each; every description is context the
        model pays for on every single turn, so the default sets are curated.
        """
        cfg = self.servers_cfg.get(server) or {}
        tool_cfg = cfg.get("tools") or {}
        include = [str(name) for name in (tool_cfg.get("include") or [])]
        exclude = {str(name) for name in (tool_cfg.get("exclude") or [])}
        cap = int(tool_cfg.get("max") or cfg.get("max_tools") or 0)

        kept = [tool for tool in tools if (not include or tool.name in include) and tool.name not in exclude]
        if cap:
            kept = kept[:cap]
        missing = [name for name in include if name not in {tool.name for tool in tools}]
        if missing:
            log.warning("mcp: %s does not offer %s", server, ", ".join(missing))
        return kept

    def _build_tool(self, server: str, info: MCPToolInfo) -> Tool | None:
        policy = policy_for(server, info.name)
        if not info.input_schema.get("properties") and info.input_schema.get("type") != "object":
            info.input_schema = {"type": "object", "properties": {}}
        schema = prune_schema(info.input_schema, self.max_description) if self.prune_schemas else info.input_schema
        needs_confirm = confirmation_for(policy, info)
        description = _short(info.description.strip() or info.title, 320)
        declared = list((schema or {}).get("properties") or {})
        if declared:
            description += " Arguments: " + ", ".join(declared[:12]) + "."
        if policy.note:
            description += " " + policy.note
        description = f"{description} (MCP: {server}.{info.name})"

        async def handler(
            ctx: ToolContext,
            args: dict[str, Any],
            _server=server,
            _tool=info.name,
            _policy=policy,
            _schema=info.input_schema,
        ) -> ToolResult:
            client = self._client(_server)
            original = dict(args or {})
            mcp_args = normalise_arguments(original, _schema)
            if mcp_args.get("__dropped__"):
                dropped = mcp_args.pop("__dropped__")
                log.info("mcp: %s.%s dropped unknown args %s", _server, _tool, dropped)
            try:
                result = await client.call_tool(_tool, _clean_args(mcp_args))
                if result.is_error:
                    # transport OK, tool refused: never let the policy summariser
                    # turn "you are not enrolled" into "no events"
                    refusal = explain_tool_error(result.text)
                    raise MCPError(
                        f"{_server}: {refusal}",
                        reason="tool_error",
                        detail=result.text[:400],
                    )
            except MCPError as exc:
                await self._mark_failure(_server, exc)
                fell_back = await self._rest_fallback(ctx, _policy, original, exc)
                if fell_back is not None:
                    if fell_back.ok:
                        # the connections card paints refusals from lastError —
                        # a call the REST twin answered is not a refusal
                        self._last_error = None
                    return fell_back
                self._last_error = {
                    "server": _server,
                    "tool": _tool,
                    "reason": exc.reason,
                    "detail": str(exc),
                    "at": time.time(),
                }
                note = f"Google refused {_server} ({_short(str(exc), 200)})"
                if note not in self._notices:
                    self._notices.append(note)
                return ToolResult(
                    ok=False,
                    error=str(exc),
                    summary=self._refusal_summary(_server, exc),
                    display={"kind": "text", "text": str(exc)[:400]},
                )
            summary, display = _policy.summarize(result, args)
            payload = result.json_payload()
            return ToolResult(
                ok=not result.is_error,
                summary=summary,
                display=display,
                data=payload if isinstance(payload, (dict, list)) else None,
                error=result.text[:500] if result.is_error else None,
            )

        def prompt(args: dict[str, Any]) -> str:
            if policy.prompt is not None:
                return policy.prompt(args)
            target = args.get("summary") or args.get("query") or args.get("subject") or ""
            verb = info.title or info.name.replace("_", " ")
            return f"Should I run {verb}?" + (f" ({target})" if target else "")

        return Tool(
            name=policy.alias,
            description=description,
            parameters=schema,
            handler=handler,
            category=policy.category,
            requires_confirmation=needs_confirm,
            confirmation_prompt=prompt,
            preview=policy.preview,
            danger=policy.danger or info.destructive,
            timeout_s=self.timeout + 10,
            # NOTE: deliberately `available` (no ctx), not `available_for` — the
            # voice turn asks is_available() *without* a ToolContext, and the
            # ctx-taking form would silently answer False there, hiding every
            # MCP tool from the model even with a working token.
            available=lambda _manager=self: _manager._token_ready,
            unavailable_hint="Connect Google in Settings → Connections to use Workspace tools.",
        )

    def _refusal_summary(self, server: str, exc: MCPError) -> str:
        """What the model should say — never conflate "refused" with "not connected"."""
        detail = str(exc)
        # content-level refusals keep Google's raw sentence in exc.detail while
        # str(exc) is already the human explanation — match on both
        haystack = f"{exc.detail or ''} {detail}".lower()
        if "developer preview" in haystack or "preview program" in haystack:
            return (
                f"Google's {server} MCP server refused this call: {detail}. "
                "IMPORTANT: the user's Google account IS connected and the tool exists. These MCP "
                "servers went generally available on 24 Nov 2026, but the rollout reaches each Cloud "
                "project over up to about ten days. Say exactly that: the account is connected; the "
                "Cloud project needs the Gmail/Calendar MCP services enabled and the OAuth client "
                "marked 'Use this client for an AI-powered agent'; retry in a few days if it persists. "
                "Do NOT tell them they have no calendar/mail access or that they are signed out."
            )
        return (
            f"The {server} MCP tool refused and returned no data: {detail}. "
            "The account is connected unless the error says otherwise; report the refusal verbatim "
            "rather than claiming there is no access."
        )

    async def _rest_fallback(
        self,
        ctx: ToolContext,
        policy: Any,
        args: dict[str, Any],
        exc: MCPError,
    ) -> ToolResult | None:
        """Run the REST twin when the MCP endpoint refuses the call.

        Google's MCP rollout (GA 24 Nov 2026) reaches each Cloud project over up
        to ~10 days; until it lands, the classic API does the same job so a
        voice request does not fail because of the rollout. The user is told
        once when the twin is needed.
        """
        if not self.allow_rest_fallback or self.fallback_lookup is None:
            return None
        twin = self.fallback_lookup(policy.alias)
        if twin is None or not twin.is_available(ctx):
            return None
        try:
            result = await twin.invoke(ctx, _rest_args(args, twin))
        except Exception as exc2:  # noqa: BLE001
            log.info("mcp: REST fallback for %s failed too (%s)", policy.alias, exc2)
            return None
        if not result.ok:
            note = _api_disabled_note(result.error or result.summary or "")
            if note:
                if note not in self._notices:
                    self._notices.append(note)
                result.summary = f"{note} {result.summary or ''}".strip()
            return result
        log.info("mcp: %s failed (%s) → answered with the REST tool", policy.alias, exc.reason)
        note = (
            f"Google's MCP endpoint refused {policy.alias} ({_short(str(exc), 160)}). "
            "Answered with the REST API instead."
        )
        if note not in self._notices:
            self._notices.append(note)
        if isinstance(result.display, dict):
            result.display = {**result.display, "via": "rest"}
        return result

    async def _mark_failure(self, server: str, exc: MCPError) -> None:
        state = self._states.get(server)
        if state is not None:
            state.ok = False
            state.reason = exc.reason
            state.detail = str(exc)

    # ------------------------------------------------------------------ access
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values() if tool.is_available()]

    def catalog(self) -> list[dict[str, Any]]:
        out = []
        for tool in sorted(self._tools.values(), key=lambda t: (t.category, t.name)):
            entry = tool.to_catalog()
            entry["source"] = "mcp"
            out.append(entry)
        return out

    def status(self) -> dict[str, Any]:
        servers = [state.to_dict(include_tools=True) for state in self._states.values()]
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "active": bool(self._tools) and self._token_ready,
            "tokenReady": self._token_ready,
            "tokenError": self._token_error,
            "authoritative": self.authoritative,
            "restFallback": self.allow_rest_fallback,
            "tools": len(self._tools),
            "servers": servers,
            "notices": self._notices[-3:],
            "lastError": self._last_error,
        }

    #: the cheapest harmless call per server, used by "Test connection"
    SELF_TEST_TOOLS = {"gmail": ("list_labels", {}), "calendar": ("list_calendars", {})}

    async def self_test(self, ctx: ToolContext | None = None) -> dict[str, Any]:
        """Actually call each server once, with the user's token.

        Discovery (`tools/list`) is anonymous, so it can look healthy while the
        real calls are blocked (client not marked as an AI agent, MCP service not
        enabled, missing scope). This makes one read-only call per server and
        reports what Google said.
        """
        await self.discover(force=True)
        results: dict[str, Any] = {}
        for name in self.server_names():
            tool_name, args = self.SELF_TEST_TOOLS.get(name, ("", {}))
            if not self._token_ready:
                results[name] = {
                    "ok": False,
                    "reason": "unauthorized",
                    "detail": self._token_error or "Connect Google first.",
                }
                continue
            if not tool_name:
                results[name] = {"ok": False, "reason": "no_probe", "detail": "no read-only probe for this server"}
                continue
            entry: dict[str, Any] = {"ok": False, "tool": tool_name, "reason": "", "detail": "", "mcp": "", "rest": ""}
            try:
                result = await self._client(name).call_tool(tool_name, args)
                if result.is_error:
                    entry["reason"] = "tool_error"
                    entry["mcp"] = explain_tool_error(result.text)
                else:
                    entry["ok"] = True
                    entry["mcp"] = "ok"
                    entry["detail"] = (result.text or "")[:400]
            except MCPError as exc:
                entry["reason"] = exc.reason
                entry["mcp"] = str(exc)
            except Exception as exc:  # noqa: BLE001
                entry["reason"] = "error"
                entry["mcp"] = f"{type(exc).__name__}: {exc}"

            # MCP refused → check the REST twin too, so one button tells the
            # whole truth ("gated by preview" *and* "Calendar API disabled")
            if not entry["ok"]:
                rest = await self._probe_rest(ctx, name, tool_name)
                if rest:
                    entry["rest"] = rest["detail"]
                    if rest["ok"]:
                        entry["ok"] = True
                        entry["reason"] = ""
                    else:
                        entry["hint"] = rest.get("hint", "")
            if not entry["detail"]:
                entry["detail"] = " · ".join(
                    part for part in (
                        f"MCP: {entry['mcp']}" if entry.get("mcp") else "",
                        f"REST: {entry['rest']}" if entry.get("rest") else "",
                    ) if part
                )[:500]
            results[name] = entry
        return {
            "ok": bool(results) and all(item.get("ok") for item in results.values()),
            "tokenReady": self._token_ready,
            "servers": results,
            "status": self.status(),
        }

    async def _probe_rest(self, ctx: ToolContext | None, server: str, tool_name: str) -> dict[str, Any] | None:
        """One read through the REST twin, to see whether *that* path is usable.

        Skipped entirely in strict MCP mode: suggesting the classic APIs to someone
        who deliberately does not want them would be noise.
        """
        if ctx is None or not self.allow_rest_fallback:
            return None
        google = getattr(ctx.integrations, "google", None)
        if google is None:
            return None
        probes = {
            "gmail": lambda: google.gmail_search("in:inbox", max_results=1),
            "calendar": lambda: google.calendar_list(max_results=1),
        }
        call = probes.get(server)
        if call is None:
            return None
        try:
            if not google.connected:
                return {"ok": False, "detail": "Google is not connected", "hint": ""}
            await asyncio.wait_for(call(), timeout=15)
            return {"ok": True, "detail": "ok", "hint": ""}
        except Exception as exc:  # noqa: BLE001
            note = _api_disabled_note(str(exc))
            return {"ok": False, "detail": note or str(exc)[:200], "hint": note}

    def prompt_note(self, send_available: bool = False) -> str:
        """Facts the persona prompt needs when MCP changes what is possible."""
        if not self._tools or not self._token_ready:
            return ""
        notes: list[str] = []
        if not send_available and not any(name.startswith("gmail_send") for name in self._tools):
            notes.append(
                "Email: you can read, search, label and draft, but you cannot send. When the user wants "
                "to send something, create the draft and tell them to hit send in Gmail."
            )
        if self.authoritative:
            notes.append("Workspace access goes through Google's MCP servers; do not claim REST access.")
        return "\n".join(notes)

    def drain_notices(self) -> list[str]:
        notices, self._notices = self._notices, []
        return notices

    async def aclose(self) -> None:
        for client in self._clients.values():
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass
        self._clients.clear()


#: properties that only matter to a form, never to a spoken request
_DROP_PROPERTIES = {
    # NOTE: overrideReminders/useDefaultReminders are deliberately KEPT — "remind
    # me ten minutes before" is a spoken request, not form bloat.
    "attachments", "conferenceData", "guestPermissions",
    "workingLocationProperties", "outOfOfficeProperties", "focusTimeProperties",
    "eventTypeFilter", "eventType", "source", "color", "colorPreset", "labelListVisibility",
    "messageListVisibility", "autoCreateParentLabels", "includeSignature", "htmlBody", "pageToken",
    "birthdayProperties", "attachment", "id", "etag", "kind",
}


def prune_schema(schema: dict[str, Any], max_description: int = 110) -> dict[str, Any]:
    """Shrink a published JSON schema to what a voice request can actually use.

    Google's Calendar `create_event` schema alone is ~9 KB because it describes
    conferences, working locations, guest permissions and reminders. Every byte
    is billed on every turn, so nested form-only properties are dropped, option
    descriptions are trimmed, and pagination is removed (nobody says "page two").
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    def walk(node: Any, depth: int = 0) -> Any:
        if isinstance(node, list):
            return [walk(item, depth) for item in node]
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key.startswith("x-"):
                continue  # vendor extension blobs (enum descriptions…) are pure bloat
            if key in ("description", "title") and isinstance(value, str):
                text = " ".join(value.split())
                out[key] = text if len(text) <= max_description else text[: max_description - 1].rstrip() + "…"
                continue
            if key == "properties" and isinstance(value, dict):
                pruned = {
                    name: walk(prop, depth + 1)
                    for name, prop in value.items()
                    if name not in _DROP_PROPERTIES and depth < 3
                }
                out[key] = pruned
                continue
            out[key] = walk(value, depth + 1)
        return out

    pruned = walk(schema)
    # required entries for dropped properties would make the tool uncallable
    if isinstance(pruned.get("required"), list):
        available = set((pruned.get("properties") or {}).keys())
        pruned["required"] = [name for name in pruned["required"] if name in available]
    return pruned


def _api_disabled_note(error: str) -> str:
    """Turn "API has not been used in project X" into an actionable line + link."""
    text = (error or "").lower()
    if "has not been used in project" not in text and "is disabled" not in text:
        return ""
    import re

    match = re.search(r"project (\d+)", error or "")
    project = match.group(1) if match else ""
    api = ""
    if "calendar" in text:
        api = "calendar-json.googleapis.com"
    elif "gmail" in text:
        api = "gmail.googleapis.com"
    if not api:
        return "The REST API for this tool is not enabled in the Google Cloud project."
    url = f"https://console.cloud.google.com/apis/library/{api}"
    if project:
        url += f"?project={project}"
    label = {"calendar-json.googleapis.com": "Google Calendar", "gmail.googleapis.com": "Gmail"}.get(api, api)
    return f"Enable the {label} API for this project, then retry: {url}"


def _short(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


#: REST-style argument names the model reaches for, mapped onto MCP's
def _rest_args(args: dict[str, Any], twin: Any) -> dict[str, Any]:
    """Fold the model's MCP-style argument names onto the REST twin's schema.

    The turn offered the MCP tool, so the model speaks `startTime`/`pageSize`;
    the classic tool declares `time_min`/`max_results`. Names the twin does not
    declare are dropped rather than poisoning the request.
    """
    declared = set(((getattr(twin, "parameters", None) or {}).get("properties") or {}))
    # one MCP name can hide behind several REST names (`startTime` is `time_min`
    # on the list tool but `start` on the create tool) — try every candidate and
    # keep the first one the twin actually declares
    candidates: dict[str, list[str]] = {}
    for rest_name, mcp_name in _ALIASES.items():
        candidates.setdefault(mcp_name, []).append(rest_name)
    out: dict[str, Any] = {}
    for key, value in (args or {}).items():
        name = str(key)
        if name in declared:
            out[name] = value
            continue
        for target in candidates.get(name, []) + candidates.get(name.lower(), []):
            if target in declared:
                out[target] = value
                break
    return out


_ALIASES = {
    "time_min": "startTime",
    "time_max": "endTime",
    "timemin": "startTime",
    "timemax": "endTime",
    "max_results": "pageSize",
    "maxresults": "pageSize",
    "limit": "pageSize",
    "query_string": "query",
    "search_query": "query",
    "event_id": "eventId",
    "eventid": "eventId",
    "message_id": "messageId",
    "thread_id": "threadId",
    "draft_id": "draftId",
    "calendar_id": "calendarId",
    "label_ids": "labelIds",
    "label_id": "labelIds",
    "attendee_emails": "attendeeEmails",
    "reminders": "overrideReminders",
    "use_default_reminders": "useDefaultReminders",
    "all_day": "allDay",
    "time_zone": "timeZone",
    "summary_text": "summary",
    "title": "summary",
    "start": "startTime",
    "end": "endTime",
    "date_time": "startTime",
    "attendees_emails": "attendeeEmails",
    "attendee_emails": "attendeeEmails",
    "attendees_list": "attendeeEmails",
}


def _coerce_value(name: str, value: Any) -> Any:
    """Shape values the model gives REST-style into what the MCP tool declares.

    Calendar MCP wants `startTime` as a plain RFC3339 string, while the REST habit
    is `{"start": {"dateTime": …, "timeZone": …}}`; attendees are email strings,
    not objects. Getting this wrong is a 400.
    """
    if name in ("startTime", "endTime") and isinstance(value, dict):
        for key in ("dateTime", "date", "time", "value"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
        return value
    if name == "attendeeEmails" and isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                email = item.get("email") or item.get("address")
                if isinstance(email, str) and email.strip():
                    out.append(email.strip())
        return out
    if name == "summary" and isinstance(value, (int, float)):
        return str(value)
    return value


def normalise_arguments(args: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Make the model's arguments fit the MCP tool's schema.

    A voice model arrives with Gmail/Calendar REST habits (`time_min`,
    `max_results`, `event_id`) while Google's MCP tools speak `startTime`,
    `pageSize`, `eventId` — and the servers reject unknown fields with a 400. So
    snake_case and common REST names are folded onto the declared properties, and
    anything the tool does not declare is dropped (and logged) instead of
    poisoning the request.
    """
    declared = (schema or {}).get("properties") or {}
    if not isinstance(args, dict) or not args:
        return {}
    lowered = {str(name).lower(): name for name in declared}
    out: dict[str, Any] = {}
    dropped: list[str] = []

    for key, value in args.items():
        if value is None or value == "" or value == [] or value == {}:
            continue
        name = key
        # `attendees` (objects) and `attendeeEmails` (strings) both exist on the
        # Calendar tools; plain email strings are the unambiguous choice
        if name == "attendees" and "attendeeEmails" in declared and isinstance(value, list):
            converted = _coerce_value("attendeeEmails", value)
            if converted:
                name, value = "attendeeEmails", converted
        if name not in declared:
            candidate = _ALIASES.get(name.lower()) or _ALIASES.get(str(name).lower().replace("-", "_"))
            if candidate and candidate in declared:
                name = candidate
            elif str(name).lower() in lowered:
                name = lowered[str(name).lower()]
            else:
                # snake_case → camelCase for anything the tool actually declares
                parts = str(name).split("_")
                camel = parts[0] + "".join(part.title() for part in parts[1:]) if len(parts) > 1 else str(name)
                if camel in declared:
                    name = camel
                elif str(name).lower() in {"starttime", "endtime"} and lowered.get(str(name).lower()):
                    name = lowered[str(name).lower()]
                else:
                    dropped.append(str(key))
                    continue
        if name == "pageSize" and isinstance(value, str) and value.isdigit():
            value = int(value)
        out[name] = _coerce_value(name, value)

    if dropped:
        out["__dropped__"] = dropped
    return out


def _clean_args(args: dict[str, Any]) -> dict[str, Any]:
    """Drop empty values so MCP servers see only what was actually given."""
    out: dict[str, Any] = {}
    for key, value in (args or {}).items():
        if value is None or value == "" or value == [] or value == {}:
            continue
        out[key] = value
    return out
