"""A small, dependency-free MCP client (streamable HTTP transport).

Just enough of the Model Context Protocol to be a *host*:

    initialize  →  notifications/initialized
    tools/list  (paginated)
    tools/call
    notifications/cancelled

It speaks JSON-RPC 2.0 over POST, accepts either a JSON body or an SSE stream as
the response, carries ``Mcp-Session-Id`` when the server issues one (the
2025-06-18 revision) and works without it too (the 2026-07-28 revision dropped
protocol-level sessions), and always sends ``MCP-Protocol-Version`` after the
handshake.

Deliberately no SDK: the app already ships httpx, the surface we need is four
methods, and owning it means we can inject an ASGI transport in tests and log
exactly what went over the wire.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx

log = logging.getLogger("otto.mcp")

#: what we ask for first; the server may answer with an older/newer revision
PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "otto", "version": "0.1.0"}


class MCPError(RuntimeError):
    """A failure the caller can explain to a human."""

    def __init__(self, message: str, *, reason: str = "error", status: int | None = None, detail: str = "") -> None:
        super().__init__(message)
        self.reason = reason
        self.status = status
        self.detail = detail


@dataclass(slots=True)
class MCPToolInfo:
    name: str
    title: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] = field(default_factory=dict)
    server: str = ""

    @property
    def read_only(self) -> bool:
        return bool(self.annotations.get("readOnlyHint"))

    @property
    def destructive(self) -> bool:
        return bool(self.annotations.get("destructiveHint"))


@dataclass(slots=True)
class MCPCallResult:
    content: list[dict[str, Any]]
    structured: Any = None
    is_error: bool = False

    @property
    def text(self) -> str:
        parts = []
        for block in self.content:
            if block.get("type") == "text" and block.get("text"):
                parts.append(str(block["text"]))
        return "\n".join(parts).strip()

    def json_payload(self) -> Any:
        """Best-effort structured value: structuredContent first, then JSON text."""
        if self.structured is not None:
            return self.structured
        text = self.text
        if not text:
            return None
        try:
            return json.loads(text)
        except ValueError:
            return None


class MCPClient:
    def __init__(
        self,
        url: str,
        *,
        name: str = "mcp",
        token_provider: Callable[[], Awaitable[str]] | None = None,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.url = url
        self.name = name
        self.token_provider = token_provider
        self.timeout = timeout
        self._transport = transport
        self._http: httpx.AsyncClient | None = None
        self._ids = itertools.count(1)
        self._session_id: str | None = None
        self._protocol_version: str = PROTOCOL_VERSION
        self._initialized = False
        self._lock = asyncio.Lock()
        self.server_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ plumbing
    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            kwargs: dict[str, Any] = {"timeout": httpx.Timeout(self.timeout, connect=10.0)}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._http = httpx.AsyncClient(**kwargs)
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        self._initialized = False
        self._session_id = None

    async def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            "user-agent": "otto-mcp/0.1",
        }
        if self.token_provider is not None:
            try:
                token = await self.token_provider()
            except Exception as exc:  # noqa: BLE001 - no token yet is not an error here:
                log.debug("%s: no access token (%s)", self.name, exc)  # tools/list is anonymous
                token = ""
            if token:
                headers["authorization"] = f"Bearer {token}"
        if self._initialized:
            headers["mcp-protocol-version"] = self._protocol_version
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    # ---------------------------------------------------------------- handshake
    async def initialize(self) -> dict[str, Any]:
        async with self._lock:
            if self._initialized:
                return self.server_info
            result = await self._rpc(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": CLIENT_INFO,
                },
                allow_reinit=False,
            )
            self._protocol_version = str(result.get("protocolVersion") or PROTOCOL_VERSION)
            self.server_info = {
                "protocolVersion": self._protocol_version,
                "serverInfo": result.get("serverInfo") or {},
                "capabilities": result.get("capabilities") or {},
            }
            self._initialized = True
            # best-effort notification; a server without sessions may answer 202/204
            try:
                await self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
            except MCPError:
                log.debug("%s: initialized notification rejected", self.name)
            log.info(
                "%s: connected (%s, protocol %s)",
                self.name,
                (self.server_info.get("serverInfo") or {}).get("name", "?"),
                self._protocol_version,
            )
            return self.server_info

    # -------------------------------------------------------------------- tools
    async def list_tools(self) -> list[MCPToolInfo]:
        await self.initialize()
        tools: list[MCPToolInfo] = []
        cursor: str | None = None
        for _ in range(20):  # hard stop on a runaway cursor
            params: dict[str, Any] = {} if cursor is None else {"cursor": cursor}
            result = await self._rpc("tools/list", params)
            for item in result.get("tools") or []:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                tools.append(
                    MCPToolInfo(
                        name=str(item["name"]),
                        title=str(item.get("title") or item["name"]),
                        description=str(item.get("description") or ""),
                        input_schema=item.get("inputSchema") or {"type": "object", "properties": {}},
                        annotations=item.get("annotations") or {},
                        server=self.name,
                    )
                )
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPCallResult:
        await self.initialize()
        result = await self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
        content = result.get("content")
        if not isinstance(content, list):
            content = [{"type": "text", "text": json.dumps(result)[:4000]}]
        return MCPCallResult(
            content=content,
            structured=result.get("structuredContent"),
            is_error=bool(result.get("isError")),
        )

    # ------------------------------------------------------------------ jsonrpc
    async def _rpc(self, method: str, params: dict[str, Any], *, allow_reinit: bool = True) -> dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params}
        try:
            body = await self._send(payload)
        except MCPError as exc:
            # a server restart/session expiry: re-handshake once, then retry
            if allow_reinit and exc.reason == "session_expired":
                log.info("%s: session expired, re-initializing", self.name)
                self._initialized = False
                self._session_id = None
                await self.initialize()
                return await self._rpc(method, params, allow_reinit=False)
            raise
        if body is None:
            raise MCPError(f"{self.name}: empty response to {method}", reason="protocol")
        if "error" in body:
            error = body["error"] or {}
            raise MCPError(
                f"{self.name}: {error.get('message') or 'JSON-RPC error'}",
                reason="rpc_error",
                detail=json.dumps(error)[:600],
            )
        result = body.get("result")
        return result if isinstance(result, dict) else {}

    async def _send(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """POST one JSON-RPC message; returns the parsed response (None for notifications)."""
        headers = await self._headers()
        is_notification = "id" not in payload
        try:
            response = await self.http.post(self.url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise MCPError(f"{self.name}: {exc}", reason="network") from exc

        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id

        if response.status_code in (401, 403):
            detail = response.text[:400]
            reason = "unauthorized" if response.status_code == 401 else "forbidden"
            raise MCPError(
                f"{self.name}: HTTP {response.status_code} — {_explain_http(detail)}",
                reason=reason,
                status=response.status_code,
                detail=detail,
            )
        if response.status_code == 404 and self._initialized:
            raise MCPError(f"{self.name}: session expired", reason="session_expired", status=404)
        if response.status_code >= 400:
            # Google's MCP servers reject a bad tool argument with HTTP 400 *and*
            # a well-formed JSON-RPC body saying exactly what was wrong. Losing
            # that behind a generic status line makes refusances unexplainable,
            # so read the body first and only fall back to the status wording.
            try:
                body = response.json()
            except ValueError:
                body = None
            if isinstance(body, dict) and "result" in body:
                return body
            if isinstance(body, dict) and body.get("error"):
                error = body["error"] or {}
                raise MCPError(
                    f"{self.name}: {error.get('message') or 'JSON-RPC error'}",
                    reason="rpc_error",
                    status=response.status_code,
                    detail=json.dumps(error)[:600],
                )
            detail = response.text[:600]
            raise MCPError(
                f"{self.name}: HTTP {response.status_code} — {_explain_http(detail)}",
                reason="http_error",
                status=response.status_code,
                detail=detail,
            )
        if is_notification or response.status_code in (202, 204) or not response.content:
            return None

        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return _parse_sse(response.text, payload.get("id"))
        try:
            return response.json()
        except ValueError:
            return _parse_sse(response.text, payload.get("id"))


def _parse_sse(text: str, request_id: Any) -> dict[str, Any] | None:
    """Pull the JSON-RPC response for ``request_id`` out of an SSE body."""
    fallback: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            message = json.loads(chunk)
        except ValueError:
            continue
        if not isinstance(message, dict):
            continue
        if "id" in message and (request_id is None or message.get("id") == request_id):
            return message
        if fallback is None and ("result" in message or "error" in message):
            fallback = message
    return fallback


def explain_tool_error(text: str) -> str:
    """A tool can *succeed transport-wise* and still report a refusal in its text.

    Google's servers do exactly that for the Developer Preview gate, so the body
    has to be read as an error rather than as an empty result.
    """
    lowered = (text or "").lower()
    if "developer preview" in lowered or "preview program" in lowered:
        return (
            "Google's Workspace MCP servers went generally available on 24 Nov 2026, but the rollout "
            "reaches each Cloud project over up to ~10 days — enable the Gmail/Calendar MCP services "
            "in the Cloud console, mark the OAuth client 'Use this client for an AI-powered agent', "
            "then try again"
        )
    if "has not been used in project" in lowered or "is disabled" in lowered:
        return "the API for this tool is not enabled in the Google Cloud project"
    if "agentic" in lowered or "ai-powered agent" in lowered:
        return "the OAuth client is not marked as an AI-powered agent"
    if "insufficient" in lowered or "scope" in lowered:
        return "the access token is missing a scope this tool needs — reconnect Google"
    return _short_text(text, 200)


def _short_text(text: str, limit: int) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _explain_http(detail: str) -> str:
    """Turn Google's HTTP error body into the sentence worth showing a human."""
    lowered = detail.lower()
    if "developer preview" in lowered or "preview program" in lowered:
        return (
            "this Workspace MCP tool is still gated behind the Google Workspace Developer Preview "
            "Program — the project has to be enrolled at developers.google.com/workspace/preview"
        )
    if "not found" in lowered or "unknown" in lowered:
        return "the MCP server did not recognise that request"
    if "invalid" in lowered or "field" in lowered:
        return "the arguments did not match what the MCP tool expects"
    return _explain_forbidden(detail)


def _explain_forbidden(detail: str) -> str:
    lowered = detail.lower()
    if "agentic" in lowered or "ai-powered agent" in lowered:
        return (
            "this OAuth client is not marked as an AI agent. In Google Cloud → Google Auth Platform → "
            "Clients, edit the client and tick 'Use this client for an AI-powered agent'."
        )
    if "has not been used" in lowered or "disabled" in lowered or "is not enabled" in lowered:
        return (
            "the MCP service is not enabled in this project. Enable gmailmcp.googleapis.com / "
            "calendarmcp.googleapis.com (APIs & Services → Library)."
        )
    if "insufficient" in lowered or "scope" in lowered:
        return "the access token is missing a scope this tool needs — reconnect Google with the MCP scopes."
    return "the MCP endpoint refused the request (see docs/MCP.md)."
