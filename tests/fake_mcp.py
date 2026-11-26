"""A tiny in-process MCP server (streamable HTTP) used by the tests.

Implements just enough of the protocol to exercise the client: initialize,
notifications/initialized, tools/list with pagination, tools/call, session ids,
SSE responses, 401/403/404 paths and JSON-RPC errors.
"""
from __future__ import annotations

import json
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route


class FakeMCPServer:
    def __init__(
        self,
        *,
        tools: list[dict[str, Any]],
        results: dict[str, Any] | None = None,
        require_token: str = "",
        forbidden_detail: str = "",
        forbid: bool = False,
        forbid_calls: bool = False,
        sse: bool = False,
        issue_session: bool = True,
    ) -> None:
        self.tools = tools
        self.results = results or {}
        self.require_token = require_token
        self.forbidden_detail = forbidden_detail
        self.forbid = forbid
        self.forbid_calls = forbid_calls
        self.sse = sse
        self.issue_session = issue_session
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.initialized = 0
        self.session_id = "sess-1234"
        self.session_expired = False
        self.protocol_versions: list[str] = []
        self.app = Starlette(
            routes=[
                Route("/mcp/v1", self.endpoint, methods=["POST"]),
                Route("/mcp/v1", self.probe, methods=["GET"]),
            ]
        )

    # ------------------------------------------------------------------- helpers
    async def probe(self, request: Request) -> Response:
        return JSONResponse({"ok": True})

    def _reply(self, payload: dict[str, Any], *, request_id: Any) -> Response:
        body = {"jsonrpc": "2.0", "id": request_id, "result": payload}
        if not self.sse:
            return JSONResponse(body)
        chunk = f"event: message\ndata: {json.dumps(body)}\n\n"
        return PlainTextResponse(chunk, media_type="text/event-stream")

    # ------------------------------------------------------------------ endpoint
    async def endpoint(self, request: Request) -> Response:
        token = request.headers.get("authorization", "")
        if self.forbid:
            return JSONResponse({"error": {"message": self.forbidden_detail or "forbidden"}}, status_code=403)
        if self.require_token and token != f"Bearer {self.require_token}":
            return JSONResponse({"error": {"message": "unauthorized"}}, status_code=401)
        if request.headers.get("mcp-protocol-version"):
            self.protocol_versions.append(request.headers["mcp-protocol-version"])

        message = json.loads(await request.body() or b"{}")
        method = message.get("method")
        request_id = message.get("id")

        if method == "initialize":
            self.initialized += 1
            self.session_expired = False
            params = message.get("params") or {}
            headers = {}
            if self.issue_session:
                headers["mcp-session-id"] = self.session_id
            body = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": params.get("protocolVersion") or "2025-06-18",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "fake-workspace", "version": "1.0"},
                },
            }
            return JSONResponse(body, headers=headers)

        if method == "notifications/initialized":
            return Response(status_code=202)

        if self.session_expired or (
            self.issue_session and request.headers.get("mcp-session-id") != self.session_id
        ):
            return JSONResponse({"error": {"message": "session not found"}}, status_code=404)

        if method == "tools/list":
            cursor = (message.get("params") or {}).get("cursor")
            page = self.tools[:2] if cursor is None else self.tools[2:]
            payload: dict[str, Any] = {"tools": page}
            if cursor is None and len(self.tools) > 2:
                payload["nextCursor"] = "page2"
            return self._reply(payload, request_id=request_id)

        if method == "tools/call":
            if self.forbid_calls:
                return JSONResponse(
                    {"error": {"message": self.forbidden_detail or "forbidden"}}, status_code=403
                )
            params = message.get("params") or {}
            name = str(params.get("name") or "")
            args = params.get("arguments") or {}
            self.calls.append((name, args))
            result = self.results.get(name)
            if callable(result):
                result = result(args)
            if result is None:
                return JSONResponse(
                    {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": f"unknown tool {name}"}}
                )
            payload = result if isinstance(result, dict) and "content" in result else {
                "content": [{"type": "text", "text": json.dumps(result)}],
                "structuredContent": result,
            }
            return self._reply(payload, request_id=request_id)

        return JSONResponse({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "no such method"}})


def gmail_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "search_threads",
            "title": "Search threads",
            "description": "Search Gmail threads.",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "pageSize": {"type": "integer"}},
            },
            "annotations": {"readOnlyHint": True},
        },
        {
            "name": "create_draft",
            "title": "Create draft",
            "description": "Create a draft email.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "to": {"type": "array", "items": {"type": "string"}},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
            },
            "annotations": {"readOnlyHint": False, "destructiveHint": False},
        },
        {
            "name": "get_message",
            "title": "Get message",
            "description": "Read one message.",
            "inputSchema": {"type": "object", "properties": {"messageId": {"type": "string"}}},
            "annotations": {"readOnlyHint": True},
        },
    ]


def calendar_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "list_events",
            "title": "List events",
            "description": "List calendar events.",
            "inputSchema": {
                "type": "object",
                "properties": {"startTime": {"type": "string"}, "endTime": {"type": "string"}},
            },
            "annotations": {"readOnlyHint": True},
        },
        {
            "name": "create_event",
            "title": "Create event",
            "description": "Create a calendar event.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "start": {"type": "object"},
                    "end": {"type": "object"},
                },
            },
            "annotations": {"readOnlyHint": False, "destructiveHint": False},
        },
        {
            "name": "update_event",
            "title": "Update event",
            "description": "Update a calendar event.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "eventId": {"type": "string"},
                    "summary": {"type": "string"},
                    "startTime": {"type": "string"},
                    "endTime": {"type": "string"},
                },
            },
            "annotations": {"readOnlyHint": False, "destructiveHint": False},
        },
        {
            "name": "delete_event",
            "title": "Delete event",
            "description": "Delete a calendar event.",
            "inputSchema": {"type": "object", "properties": {"eventId": {"type": "string"}}},
            "annotations": {"readOnlyHint": False, "destructiveHint": True},
        },
    ]
