# MCP: Google Workspace access for an AI agent

Google now requires AI agents to reach Workspace **through their MCP servers**. The REST endpoints
still work for ordinary apps, but from **1 February 2027** agent traffic is expected to go through
MCP. Otto's Gmail and Calendar tools are therefore MCP-backed by default; the old REST
implementations are still in the tree as a fallback for accounts whose project cannot use MCP yet.

```
browser ─▶ Otto harness ─▶ otto/mcp/manager.py ──JSON-RPC──▶ gmailmcp.googleapis.com/mcp/v1self
                                 │                            (and calendarmcp.…)
                                 └─ per-user Google access token (Authorization: Bearer)
```

## What the servers expose

| Server | Endpoint | Tools in the default set |
|---|---|---|
| Gmail MCP | `https://gmailmcp.googleapis.com/mcp/v1self` | `search_threads`, `get_thread`, `get_message`, `create_draft`, `list_drafts`, `list_labels`, `label_message`, `unlabel_message` |
| Calendar MCP | `https://calendarmcp.googleapis.com/mcp/v1self` | `list_events`, `search_events`, `get_event`, `list_calendars`, `suggest_time`, `create_event`, `update_event`, `delete_event`, `respond_to_event` |

The servers publish more than this — Gmail has **23** tools (trash, spam, sensitive labels, label
CRUD…) and Calendar 9. Every tool description is paid for on every turn, so the default set is
curated in `config/app.yaml → mcp.servers.<name>.tools.include`; add names there (or set
`include: []` for everything) to widen it. Aliases and confirmation policies exist for all 23 Gmail
tools, so widening the list needs no code change.

Schemas are also **pruned** before they reach the model (`mcp.prune_schemas`): form-only properties
(attachments, conference data, guest permissions, reminders, pagination) are dropped and descriptions
are trimmed. Google's Calendar `create_event` schema is ~9 KB verbatim; pruning takes the whole set
from ~18k to ~10k tokens, and the curated set brings it to roughly 6–7k.

Two consequences worth knowing:

* **There is no `send` tool yet.** Otto can read, search, label and *draft* mail; sending still goes
  through the REST API while it exists. When the draft is ready Otto says so and you press send. The
  persona prompt is adjusted automatically (`manager.prompt_note()`), so the model never promises a
  send it cannot do.
* **Tool names are aliased.** `search_threads` becomes `gmail_search` and `create_event` becomes
  `calendar_create_event`, so existing prompts, the entity memory ("move *that* to four") and the UI
  cards are unchanged. Anything without an alias shows up as `<server>_<tool>`.

## Configure your Google Cloud project

Everything happens on the client you already made for the login, plus two flags:

0. **Enable the classic APIs too.** The MCP servers are a *front door*; the REST fallback (and
   everything non-agent) still needs the underlying APIs. In the same project enable
   **Gmail API** (`gmail.googleapis.com`) **and Google Calendar API** (`calendar-json.googleapis.com`).
   With only the two MCP services enabled, every MCP call that Google gates (see below) falls back to
   a REST call that then fails with *"API has not been used in project …"* — Otto now prints the
   enable link when that happens.
1. **Enable the MCP services** — *APIs & Services → Library* → search and enable:
   * **Gmail MCP API** (`gmailmcp.googleapis.com`)
   * **Calendar MCP API** (`calendarmcp.googleapis.com`)

   ![the two services appear under "Gmail MCP API" / "Calendar MCP API"]
2. **Mark the OAuth client as an AI agent** — *Google Auth Platform → Clients* → open your client →
   tick **“Use this client for an AI-powered agent”** → Save. Google blocks clients without this
   setting from every Workspace MCP endpoint. (When you create a client through
   *Personal mode onboarding* the box is ticked for you.)
3. **Scopes** — Otto asks for Google's documented MCP scope set: `gmail.readonly`,
   `gmail.compose`, `gmail.modify`, `calendar.calendarlist.readonly`, `calendar.events.freebusy`,
   `calendar.events.readonly`, `calendar.events`, `userinfo.email`. Declare every one on the
   *Data Access* page of the consent screen — an undeclared scope fails consent with
   `invalid_scope`. They live in `config/app.yaml → google.scopes`.
4. **Reconnect Google** in Settings → Connections (the consent screen runs again so the new client
   metadata and scopes apply).

Usage tiers and quotas now apply per call for MCP traffic as well — see
[Google's quota page](https://developers.google.com/workspace/guides/quota-usage-tiers). Otto makes
exactly one MCP call per tool call and caches `tools/list` per conversation, so it stays inside the
cheap tier for a single user.

## Strict MCP (the default): no classic APIs

`allow_rest_fallback: true` (the current config) keeps MCP first and answers refused
calls over the classic APIs while the GA rollout lands. To drop the legacy REST APIs
entirely, set it to `false` — the one line to change. Then:

* the Gmail/Calendar REST tools are unregistered — the model never sees them and cannot reach them,
* `config/app.yaml → google.scopes` asks for exactly the scopes Google documents for its MCP
  servers (`gmail.send` is gone: it existed solely for the REST send path, and Gmail's MCP server
  has no send tool),
* **Test connection** reports MCP only, with no hint about enabling product APIs,
* a blocked call fails honestly, quoting exactly what Google answered — it never becomes
  *"you have no calendar access"*.

### The gate, post-GA

The Workspace MCP servers went **generally available on 24 Nov 2026** — the old Developer Preview
enrolment is obsolete. A project that still sees *"enrolled in the … Developer Preview Program"* is
waiting for the GA rollout, which Google allows **up to ~10 days** to reach a project. Once the MCP
services are enabled and the OAuth client is marked as an AI agent (previous section), that error
means: retry later — nothing in Otto needs to change.

Otto talks to `/mcp/v1self`, Google's endpoint for MCP hosts without form-elicitation support (Otto
has its own confirmation flow). Switch the URLs to `/mcp/v1` if you ever implement elicitation.

## Verifying the wiring

Discovery is anonymous — `tools/list` succeeds even when calls are blocked — so a healthy-looking
tool count proves nothing on its own. Two checks do:

* **Settings → Connections → Workspace MCP → Test connection** (or
  `POST /api/mcp/selftest`) makes one real read call per server with your token
  (`list_labels` for Gmail, `list_calendars` for Calendar) and prints exactly what Google answered.
* **`Check again`** (`POST /api/mcp/probe`) only re-lists tools.

A passing self-test looks like:

```json
{"ok": true, "tokenReady": true,
 "servers": {"gmail": {"ok": true, "tool": "list_labels", "detail": "INBOX, Receipts, …"},
             "calendar": {"ok": true, "tool": "list_calendars", "detail": "primary"}}}
```

If it fails, the `reason`/`detail` pair names the cause: `forbidden` + *"AI-powered agent"* → tick
that box on the OAuth client; `forbidden` + *"MCP service is not enabled"* → enable
`gmailmcp.googleapis.com` / `calendarmcp.googleapis.com`; `unauthorized` → reconnect Google;
`tool_error` → look at the tool's message.

## Settings

`config/app.yaml`:

```yaml
mcp:
  enabled: true
  mode: mcp                 # mcp | auto | rest
  allow_rest_fallback: false  # default: MCP only, no legacy REST anywhere
  timeout_s: 30
  discover_timeout_s: 8
  tool_cache_s: 600
  servers:
    gmail:    { url: https://gmailmcp.googleapis.com/mcp/v1self }
    calendar: { url: https://calendarmcp.googleapis.com/mcp/v1self }
```

or the env overrides `OTTO_MCP_ENABLED` / `OTTO_MCP_MODE`.

| mode | behaviour |
|---|---|
| `mcp` (default) + `allow_rest_fallback: false` | **MCP only.** The Gmail/Calendar REST tools are not registered at all — not in the model's toolbelt, not in the UI catalog. A refused MCP call fails with Google's reason. |
| `mcp` + `allow_rest_fallback: true` | MCP first; a refused call is answered by the REST twin (needs the classic APIs enabled). |
| `auto` | same, but nothing is hidden — first tool to claim a name wins, MCP first. |
| `rest` | MCP is ignored entirely (pre-Feb-2027 behaviour). |

Discovery is **anonymous** (`tools/list` needs no token) and *use* is gated on the signed-in account's
token, so the tool list is populated before a user connects Google while the model only ever sees
tools it can actually call.

## Adding another MCP server

Any streamable-HTTP MCP endpoint can join the same table:

```yaml
mcp:
  servers:
    linear:
      url: https://mcp.linear.app/mcp
      # a per-user bearer token, if the server needs one:
      # tokenEnv: LINEAR_TOKEN
```

Its tools appear to the model as `linear_<tool>` (or with a hand-written alias/policy in
`otto/mcp/policy.py` — that is where confirmation prompts, spoken summaries and UI card kinds live
for the Workspace tools). Tools whose MCP annotations say `destructiveHint: true` always go through
Otto's spoken confirmation.

## When a tool is gated (Developer Preview, IAM, agentic client)

Google still gates some Workspace MCP tools — Calendar's, at the time of writing — behind the
**Google Workspace Developer Preview Program**. A call then fails with
`HTTP 400 … requires that your Google Cloud project (…) is enrolled`. Otto handles that in two ways:

1. **It tells you the real reason** — the MCP error keeps Google's own sentence, so the tool card and
   the assistant both see *"still gated behind the Developer Preview Program"* instead of `HTTP 400`.
2. **It falls back per call.** With `mcp.allow_rest_fallback: true` (default) the equivalent REST tool
   answers that one call, the result is tagged `"via": "rest"`, and Otto says once per conversation
   that it used the REST API. Reads *and* writes fall back, so nothing in the conversation breaks
   while your project waits for enrolment. Set it to `false` once every server you use is GA if you
   want failures to be loud instead.

Argument names are normalised before the call for the same reason: a voice model arrives with Gmail
and Calendar REST habits (`time_min`, `max_results`, `event_id`) while Google's MCP tools speak
`startTime`, `pageSize`, `eventId` — and the servers reject unknown fields with a 400. Otto folds
snake_case and the common REST aliases onto the declared properties and drops anything the tool does
not declare. Every MCP tool's description also lists its argument names, which cut this class of
error to near zero in testing.

## Troubleshooting

| what the Connections card says | meaning |
|---|---|
| `gmail: forbidden … “AI-powered agent”` | tick *Use this client for an AI-powered agent* on the OAuth client, then reconnect. |
| `gmail: forbidden … MCP service is not enabled` | enable `gmailmcp.googleapis.com` / `calendarmcp.googleapis.com` in the project. |
| `Not connected` with 0 tools, badge “Not connected” | you haven't connected Google for this account yet — that's the token gate, not a failure. |
| `12 tools` but the model never calls them | the tools are there; check the model picked them (debug mode shows tool calls) or that the account's token has the scopes. |
| Everything 401s mid-conversation | the refresh token was revoked — reconnect Google. |

`POST /api/mcp/probe` (the *Check again* button) re-runs discovery and returns the raw per-server
state, which is the fastest way to see what Google answered.
