"""Google People API tools — the user's own name, and contact lookup.

Drafting email needs two things a calendar-and-mail-only setup cannot give:
who the user is (so drafts can be signed) and who the user knows (so "email
David" can resolve to an address without interrogating the user).
"""
from __future__ import annotations

import logging
from typing import Any

from .base import Tool, ToolContext, ToolResult

log = logging.getLogger("otto.tools.people")


def _google(ctx: ToolContext):
    client = ctx.integrations.google
    if not client.connected:
        raise RuntimeError("Google is not connected for this account")
    return client


def _users(ctx: ToolContext):
    integrations = ctx.integrations
    return getattr(integrations, "users", None), getattr(integrations, "user_id", None)


async def _profile(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    profile = await _google(ctx).people_profile()
    name = profile.get("name") or ""
    # remember it: the next session can sign drafts without a tool call
    users, user_id = _users(ctx)
    if name and users is not None and user_id:
        try:
            users.set_secret(user_id, "profile_name", name)
        except Exception as exc:  # noqa: BLE001 — caching must never fail the call
            log.info("could not cache profile name: %s", exc)
    if not name:
        return ToolResult(
            ok=False,
            error="the People API returned no name for this account",
            summary="No name is set on the Google profile.",
        )
    return ToolResult(
        ok=True,
        summary=f"The user's name is {name} (email {profile.get('email') or 'unknown'}). "
        "Sign drafted email with this name.",
        display={"kind": "text", "text": f"User: {name} <{profile.get('email', '')}>"},
    )


async def _search(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    query = str(args.get("query") or "").strip()
    if not query:
        return ToolResult(ok=False, error="a search query (a person's name) is required")
    matches = await _google(ctx).people_search(query)
    if not matches:
        return ToolResult(
            ok=True,
            summary=f"No contact matched '{query}'. Ask the user for the person's name or email.",
            display={"kind": "contact_list", "items": []},
        )
    lines = [f"- {m['name']} <{m['email']}>" for m in matches if m.get("email")]
    return ToolResult(
        ok=True,
        summary="; ".join(lines) if lines else f"Contact '{matches[0]['name']}' has no email address.",
        display={"kind": "contact_list", "items": matches[:5]},
    )


def _ready(ctx: ToolContext) -> bool:
    try:
        return bool(ctx.integrations.google.connected)
    except Exception:  # noqa: BLE001
        return False


TOOLS: list[Tool] = [
    Tool(
        name="people_profile",
        description=(
            "The user's own name and email, from their Google profile. Call this before drafting "
            "an email so you can sign the body with the user's name."
        ),
        parameters={"type": "object", "properties": {}},
        handler=_profile,
        category="people",
        available_for=_ready,
        unavailable_hint="Connect Google (with the People API enabled) to use this.",
    ),
    Tool(
        name="people_search",
        description=(
            "Search the user's Google contacts by name; returns the best matching name and email "
            "address. Use it whenever the user mentions a person without giving an address."
        ),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The person's name to find."}},
            "required": ["query"],
        },
        handler=_search,
        category="people",
        available_for=_ready,
        unavailable_hint="Connect Google (with the People API enabled) to use this.",
    ),
]
