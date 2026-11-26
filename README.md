# Otto

A voice assistant you can actually interrupt.

Otto listens continuously, answers out loud in a cloned-quality local voice, and uses real
tools — Gmail, Google Calendar, Slack — in the middle of a sentence when it needs them.
It is a *harness*: the model, the voice, the persona and the toolset are all swappable.

```
                        ┌──────────────────────────────────────────────┐
   🎙 mic ──16 kHz PCM──▶│  FastAPI  /ws                                │
   🔊 spk ◀─24 kHz PCM── │  ┌────────────────────────────────────────┐  │
                        │  │ Session harness (otto/session.py)      │  │
                        │  │  VAD turns → STT → model loop → TTS    │  │
                        │  │  tool calls · confirmations · barge-in │  │
                        │  └───────┬───────────────┬────────────────┘  │
                        └──────────┼───────────────┼───────────────────┘
                    Lunaroute      │               │  local GPU 1
              ┌────────────────────▼──┐   ┌────────▼─────────────────┐
              │ parakeet-tdt-0.6b-v3  │   │ Qwen3-TTS-12Hz-1.7B      │
              │ deepseek-4.1-flash    │   │ CustomVoice, streaming   │
              │ (any model you pick)  │   │ ~130 ms to first audio   │
              └───────────────────────┘   └──────────────────────────┘
```

## Why it feels like a conversation

| Behaviour | How it is implemented |
|---|---|
| **Natural turn-taking** | The browser runs an energy VAD with an adaptive noise floor; it sends `utterance.start` / `utterance.end` around real speech and streams 16 kHz PCM in between. The server transcribes only complete utterances (`parakeet-tdt-0.6b-v3`), so there are no half-word hypotheses in the model's context. |
| **Barge-in** | The mic stays live while Otto talks. Speech onset → the browser stops playback *synchronously* and sends `barge_in` → the server aborts the LLM stream, cancels queued TTS segments, flushes the client audio buffer (`cancel.audio`), and goes back to listening — typically within ~100 ms. |
| **Low latency** | Streaming everywhere: model tokens → sentence segmentation → per-sentence TTS → per-chunk PCM pushed the instant it exists. The first chunk is cut aggressively (comma, or ~60 chars) so audio starts before the first sentence is finished. Local voice: **~180–290 ms** to first audio, RTF ≈ 0.42 on an RTX 5060 Ti. A short hedge ("Mm.") covers a model that is slow to start, so silence never stretches past ~1.3 s. |
| **Tools live** | Tools are called mid-turn; the model can speak, call a tool, and keep talking. If a tool is slower than 250 ms and nothing has been said yet, Otto speaks a human filler ("Let me check.") instead of going silent. |
| **Confirmation** | Sending, posting, creating and deleting never happen silently. The tool call is parked, the question is *spoken*, a card appears in the UI, and the answer can be a click or the words "yes, go ahead". |
| **Remembers what it just did** | Every id-bearing tool result is also written to a small entity memory; the freshest events/emails/drafts are re-injected into the system prompt each turn. "What's on my calendar?" → "move **that** to four" works, and `calendar_update_event` can also resolve a *hint* ("the dentist") when the model passes no id. |
| **Background work** | Long jobs run detached (`start_background_task`) while the conversation continues, stream status to the UI, and then **announce the result out loud** — immediately if Otto is idle, or as soon as the current turn ends. |
| **Parallel subagents** | A **toggle** lets Otto split a request into up to 6 independent questions, run them in parallel private loops (read-only tools, no confirmations), and read only the conclusions. |
| **Smooth sentence joins** | Queued sentences are merged into one synthesis request, sub-40-char fragments are never sent alone, and the TTS service trims the leading/trailing silence each segment carries. No more reset-flavoured gaps between sentences. |

## The interface

Audio-mode first: one orb that breathes when idle, follows your voice when listening, and reacts to
the reply while speaking. Under it a single state line; the transcript appears as it grows — your
turns as quiet bubbles, Otto's as plain text (no bubbles, no avatars, no timestamps). Tool calls,
confirmations and background jobs are the only things with a surface, and they are 1 px bordered and
muted, not colourful cards.

* **Top bar** — wordmark + live dot, connection pill, account avatar (email, Settings, Connections,
  Sign out).
* **Composer** — pill at the bottom: mic (becomes Stop while speaking), text field, send arrow.
  `Space` hold-to-talk, `Esc` stop, `⌘K` settings.
* **Settings** — a right-hand sheet: Account, Connections, Conversation, Voice, Listening,
  Capabilities, Advanced. The same sheet is where Google and Slack are connected.
* **Login** (`/login`) — same language: wordmark, one 360 px column, inline server errors, quiet red.

Everything is one dark theme, system font stack, 150–220 ms eased transitions, and
`prefers-reduced-motion` respected.

## Measured on this box (RTX 5060 Ti + Lunaroute)

| hop | typical | notes |
|---|---|---|
| STT `whisper-large-v3` | 0.5–0.9 s | ~2× faster round-trip than parakeet through this gateway |
| LLM first token (`deepseek-4.1-flash`, `reasoning_effort: low`) | 0.3–7.5 s | gateway variance; median ≈ 2.8 s, reasoning tokens 0–47 |
| first thing the user *hears* | ~1.3 s | hedge filler at 1.1 s + 200 ms TTS, regardless of model latency |
| TTS first audio (local, warm) | 180–290 ms | RTF ≈ 0.42, so it speaks ~2.3× faster than playback |
| barge-in → silence | < 100 ms | local flush, then the server cancels the LLM + GPU work |

Practical consequences:

* Transcription is the most reliable place to win time: `whisper-large-v3` round-trips in
  ~0.5–0.9 s versus ~1.4–2.3 s for parakeet on this gateway. Whisper is the default now;
  `GET /api/bootstrap` exposes every option under `sttModels` and the session accepts
  `stt_model` in its config.
* Model latency is dominated by the gateway, not by the model: the same prompt came back in 0.3 s
  and in 7.5 s. Keep the hedge fillers (`turn_taking.first_token_filler_ms: 1100`) — they are what
  guarantees the user hears something at ~1.3 s every time.
* `deepseek-4.1-flash` runs with `reasoning_effort: low` (`config/app.yaml →
  lunaroute.model_options`). Raise it to `medium` if you want more deliberate multi-step tool use,
  or switch the block to `chat_template_kwargs: {enable_thinking: false}` for the absolute lowest
  latency.

## Quick start

```bash
cp .env.example .env          # add LUNAROUTE_API_KEY (+ Slack/Google if you want those tools)
make setup                    # app venv + GPU TTS venv (torch cu128 + faster-qwen3-tts)
make dev                      # starts Qwen3-TTS on GPU 1 and Otto on :8800
```

Open **http://127.0.0.1:8800**, create an account (email + password, stored locally in
`data/otto.db`), allow the microphone, and talk. Gmail/Calendar/Slack are connected per account in
**Settings → Connections** — “Connect Google” is a normal Google consent flow, so each user's mail
stays theirs.

Requirements: Python 3.12, `uv`, ffmpeg, a CUDA GPU (the TTS service pins itself to
`CUDA_VISIBLE_DEVICES=1` by default — edit `OTTO_TTS_GPU` to change that).

Run the two services separately if you prefer:

```bash
make tts        # local Qwen3-TTS  → http://127.0.0.1:8801
make server     # Otto harness     → http://127.0.0.1:8800
```

## How a turn works

1. **Speech in.** Browser VAD opens an utterance, streams PCM16@16 kHz, closes it after ~550 ms
   of silence. (While the user talks, the server also emits best-effort `stt.partial` captions.)
2. **Transcribe.** Lunaroute `parakeet-tdt-0.6b-v3`, ~150 ms for a sentence. Utterances shorter
   than 220 ms are discarded.
3. **Think & speak.** One streaming chat-completions call to `deepseek-4.1-flash` with the tool
   schemas attached. Text deltas go two places at once: the transcript in the browser, and a
   sentence buffer feeding the TTS queue.
4. **Tools.** A `tool_calls` delta pauses the text stream. Non-interactive tools run immediately
   (a filler phrase covers any wait); interactive ones are parked for confirmation. Results are
   appended to the context and the model continues — up to `max_tool_iterations` rounds.
5. **Speak and settle.** Each TTS segment is bracketed by `audio.start`/`audio.end`; the browser
   schedules them gaplessly. "Turn over" means *audio finished playing*, which the browser reports
   with `playback.idle`.

Barge-in, cancel and confirmation answers short-circuit this loop at any point.

## Workspace tools run on MCP

Gmail and Calendar are **MCP tools now**, not hand-rolled REST calls: Google requires AI agents to
reach Workspace through its MCP servers (agent access to the REST APIs closes on 1 Feb 2027).

* `otto/mcp/` is a small MCP host — streamable-HTTP JSON-RPC client (initialize → `tools/list` →
  `tools/call`, session handling, SSE or JSON responses), a per-user tool manager, and a policy layer
  that maps Google's tool names onto Otto's (`search_threads` → `gmail_search`,
  `create_event` → `calendar_create_event`) so the personas, the entity memory and the UI cards did
  not have to change.
* Confirmation comes from Google's own annotations (`destructiveHint` → spoken confirmation + red
  card), plus explicit policies for calendar writes.
* The tool list is discovered anonymously and *use* is gated on the signed-in account's token, so the
  UI can show “12 tools” before you connect Google, while the model only ever sees callable tools.
* **Gmail has no `send` tool on MCP yet** — Otto reads, searches, labels and drafts; sending still
  uses the REST API while it is available, and the persona prompt says so out loud.
* **This build is MCP-only by default** (`mcp.allow_rest_fallback: false`): the Gmail/Calendar REST
  tools are not registered at all, and the scope list drops `gmail.send` (that existed only for the
  REST send path). A refused MCP call fails with Google's own reason instead of silently using REST.
  Set `allow_rest_fallback: true` if you want the old REST tools back as a safety net.
* Workspace MCP servers require the Cloud project to be registered in the
  [Google Workspace Developer Preview Program](https://developers.google.com/workspace/preview)
  (a couple of days, one form, needs your Workspace account + project number).
* Full setup (enable the two MCP services, tick *“Use this client for an AI-powered agent”*, scopes,
  quotas, adding other MCP servers): **[docs/MCP.md](docs/MCP.md)**.

## Ending a conversation: the dashboard

Press **End** in the top bar (or `Shift+Esc`) and Otto hangs up and takes you to
**`/dashboard`** — the “after the call” view:

* **Actions required** — everything a tool produced that *you* have to finish. Today that is email
  drafts: Otto can draft but (on Google's MCP servers) cannot send, so each draft lands here with the
  subject, the recipients, a two-line preview and **Open in Gmail ↗**, which deep-links straight into
  the draft (`#drafts?compose=<id>`, or the draft list when the id is unknown). *Mark done* and
  *Dismiss* clear the card; finished items move to a collapsible *Recently done* list.
* Actions are **persisted per account** in `data/otto.db`, so closing the tab mid-call loses nothing,
  and the live app shows a quiet `N to finish` pill next to your avatar the moment a draft is created
  (`action.created` over the websocket).
* The page is auth-gated like everything else and keeps the same visual language: one column, no
  chrome, no orb.

## Accounts and “Connect Google”

Otto has real users now: a login page, scrypt-hashed passwords, opaque session cookies, and — the
part that matters for tools — **per-user credentials**. Gmail, Calendar and Slack act as the
signed-in account, so two people on one server never see each other's mail.

* First visit → `/login`. Sign up and you're in. Everything lives in `data/otto.db`.
* **Settings → Connections** shows the state of each integration for *your* account:
  * **Connect Google** → consent screen → back to Otto. The Gmail/Calendar tools then appear for
    you (and only you) because availability is resolved per session, not process-wide.
  * **Slack** takes a bot token, validated with one API call before it is stored.
  * **Disconnect** deletes the stored refresh token and the tools disappear immediately.
* The server needs one OAuth *client* (a JSON file or `GOOGLE_CLIENT_ID/SECRET`); each user brings
  their own consent. Click-by-click setup, the exact redirect URI, and the error table live in
  **[docs/AUTH.md](docs/AUTH.md)**.
* The old single-user CLI flow (`python scripts/google_auth.py`) still works for scripts and the
  smoke test; the web app ignores it for signed-in users.

## Background work, subagents and the two toggles

Otto can keep working while it keeps talking. Both behaviours are visible in the settings drawer.

**Background jobs** (always available). A tool named `start_background_task` puts a subagent on a
detached job and returns immediately. The job streams `job.update` events into the UI (a card with a
live status pill and a cancel button, plus a compact "running" strip above the composer). When it
finishes, the result is turned into one spoken turn — "the invoice check came back: two are
overdue" — but never on top of the user: if a turn is in progress or a confirmation is pending, the
announcement waits. `list_tasks` lets the model read results back on request.

**Parallel subagents** (*Capabilities → Parallel subagents*, off by default). When enabled, the
model gets `delegate_task`: it can split a request into up to 6 independent sub-questions, each run
in its own private loop with read-only tools (Gmail search/read, calendar read, Slack read, clock,
notes) and a small step budget, all in parallel. The main conversation then reads the conclusions
and answers — the split never reaches the transcript, so the voice stays one voice.

**Background model** (*Capabilities → Background model*). When Lunaroute publishes a
`<model>-background` sibling (`deepseek-4.1-flash-background`, `glm-5.3-flash-background`, …) this
routes subagents and background jobs there and leaves the faster model for the live conversation.
The toggle disables itself when the selected model has no such variant — `GET /api/bootstrap`
reports `capabilities.background_model = {available, id}`.

Both toggles are per-session, persisted in the browser and in `data/config.json`, and they change
what the model can *see*: with subagents off, `delegate_task` is not in the tool schema at all, so
the model cannot hallucinate a fan-out it does not have.

## Conversation memory (why "that event" works)

Every tool result that carries ids (emails, events, drafts, Slack messages) is recorded in
`otto/memory.py::EntityMemory`. At the start of each turn the freshest handful are appended to the
system prompt:

```
Items seen earlier in this conversation (ids are real — use them when the user refers back…):
- event id=evt_a "Design sync" — 2026-11-25 10:00–10:30
- email id=18f2c "Lease renewal" — from sarah@acme.com Mon
```

So a turn can span as many tool calls as it likes, history can be trimmed, and *"move that to
four"* still resolves. The calendar tools additionally accept a `hint` (part of the title) and
resolve it server-side against the same memory, so the model has a second way to hit the right row.

## Personalities

A personality is one YAML file — the whole "who" of the assistant:

```yaml
# config/personalities/otto.yaml
id: otto
name: Otto
voice:
  voice: Aiden                 # preset from the local Qwen3-TTS CustomVoice model
  instruct: Warm and friendly American male voice, relaxed pace, smiling tone.
llm:
  model: deepseek-4.1-flash    # any Lunaroute model can be picked in the UI
  temperature: 0.7
speech:
  fillers: ["One sec.", "Let me check."]
  tool_filler: "Let me check that for you."
system_prompt: |
  You are Otto ...
```

Edit it in the UI (settings → personality → *edit*) and it is saved to
`data/personalities/<id>.yaml`, shadowing the built-in file. Ship a new persona by dropping
another YAML into `config/personalities/` — it shows up in the UI on restart. Two are included:
`otto` (warm, dry, quick) and `sable` (calm and minimal).

`config/app.yaml → session.global_rules` is appended to *every* persona: the "you are being read
aloud, never use markdown" rules live there, not in each personality.

## Tools

Included out of the box (`otto/tools/`):

| Tool | Confirmation |
|---|---|
| `gmail_search`, `gmail_read`, `gmail_thread`, `gmail_labels`, `gmail_list_drafts` | – (MCP) |
| `gmail_draft` | – (MCP) |
| `gmail_label`, `gmail_unlabel` | – (MCP) |
| `gmail_send` | ✅ spoken + card (REST — MCP has no send tool yet) |
| `calendar_list_events`, `calendar_search_events`, `calendar_get_event`, `calendar_list_calendars`, `calendar_suggest_time` | – (MCP) |
| `calendar_create_event`, `calendar_update_event`, `calendar_delete_event`, `calendar_respond_to_event` | ✅ (delete is marked dangerous) |
| `slack_list_channels`, `slack_read_channel`, `slack_search` | – |
| `slack_send_message` | ✅ |
| `get_current_time`, `remember`, `recall` | – |
| `start_background_task`, `list_tasks` | – (detached work) |
| `delegate_task` | – (only when *Parallel subagents* is on) |

Unconfigured integrations are hidden from the model automatically (no phantom tool calls), and
the UI shows them as "not connected" in the tool list.

### Adding a tool

Create `otto/tools/my_thing.py`:

```python
from .base import Tool, ToolContext, ToolResult

async def handler(ctx: ToolContext, args: dict) -> ToolResult:
    return ToolResult(
        ok=True,
        summary="what the model should be told",      # short, factual
        display={"kind": "text", "text": "shown in the UI card"},
    )

TOOLS = [
    Tool(
        name="my_thing",
        description="One sentence the model uses to decide when to call it.",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        handler=handler,
        category="custom",
        # requires_confirmation=True,           # or a predicate: lambda args: args["dry_run"] is False
        # confirmation_prompt=lambda args: "…?", preview=lambda args: {...},
        # timeout_s=30, available=lambda: bool(os.environ.get("MY_KEY")),
    )
]
```

Restart the server. That is the entire extension mechanism — no registration call, no config.
Out-of-tree tools also work: point `OTTO_TOOL_PLUGINS=/path/to/tool.py,/path/to/other.py` at files
that expose the same `TOOLS` list.

## Configuration

`.env` (secrets and machine-specific values):

| Variable | Default | Meaning |
|---|---|---|
| `LUNAROUTE_API_KEY` | – | required for STT + the model |
| `LUNAROUTE_BASE_URL` | `https://gw.lunaroute.com/v1` | OpenAI-compatible gateway |
| `OTTO_STT_MODEL` | `parakeet-tdt-0.6b-v3` | transcription model |
| `OTTO_LLM_MODEL` | `deepseek-4.1-flash` | default model (UI can switch) |
| `OTTO_TTS_URL` | `http://127.0.0.1:8801` | local TTS service |
| `OTTO_TTS_ENABLED` | `true` | `false` → text-only mode |
| `OTTO_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | local voice model |
| `OTTO_TTS_GPU` | `1` | `CUDA_VISIBLE_DEVICES` for the TTS service |
| `OTTO_TTS_TRIM_SILENCE` | `1` | trim the silence Qwen3-TTS puts around each segment |
| `OTTO_PARALLEL_AGENTS` | `false` | default for the *Parallel subagents* toggle |
| `OTTO_DB_PATH` | `data/otto.db` | users, sessions, per-user connection tokens |
| `OTTO_PUBLIC_URL` | `http://127.0.0.1:8800` | base URL for links + the Google OAuth redirect |
| `GOOGLE_CLIENT_SECRETS` / `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET` | `secrets/google_client.json` / – | the one OAuth client users sign in with |
| `OTTO_BACKGROUND_MODEL` | `false` | default for the *Background model* toggle |
| `GOOGLE_CLIENT_SECRETS` / `GOOGLE_TOKEN_PATH` | `secrets/google_client.json` / `secrets/google_token.json` | Google OAuth |
| `SLACK_BOT_TOKEN` | – | Slack bot token |

`config/app.yaml` (behaviour): model list, timeouts, STT partials, TTS chunk sizes, endpoint
silence, tool-iteration cap, history length, and the global speech rules. Everything is commented.

## Google (Gmail + Calendar)

1. Google Cloud Console → new project → enable **Gmail API** and **Google Calendar API**.
2. OAuth consent screen → add yourself as a test user.
3. Credentials → **Create credentials → OAuth client ID → Desktop app** → download the JSON to
   `secrets/google_client.json`.
4. `make google-auth` — a browser opens, you consent, the token lands in `secrets/google_token.json`.

Scopes requested: `gmail.modify`, `gmail.send`, `calendar.events`, `userinfo.email`.

## Slack

Create a Slack app → Bot token with `channels:history`, `channels:read`, `groups:history`,
`chat:write`, `im:history`, `search:read`, `users:read`, `users:read.email` → invite the bot to the
channels you care about → put the `xoxb-…` token in `.env`.

## Testing

```bash
make test                     # 67 unit tests: segmentation, protocol, tools, API, full harness with fakes
make smoke                    # end-to-end against a running server: answer + tool + barge-in
./.venv/bin/python scripts/voice_check.py --voice Aiden
                              # synth → transcribe → compare, to audition the voice
```

`make smoke` signs in with `OTTO_SMOKE_EMAIL`/`OTTO_SMOKE_PASSWORD` if they are set, otherwise it
creates a throwaway account and deletes it afterwards.

`scripts/voice_check.py` is the fastest way to judge a voice or an `instruct` prompt: it speaks a
few sentences, measures time-to-first-byte and RTF, then transcribes the result with parakeet and
prints the match score, so "sounds human" and "is intelligible" are both checked in one command.

`tests/test_session.py` runs the real harness against fake STT/LLM/TTS, so turn-taking,
tool loops, confirmations, fillers, barge-in and history trimming are all covered without a GPU or
API key.

## Layout

```
otto/
  main.py            FastAPI app: auth, /ws, REST, static UI
  db.py / users.py   SQLite: users, sessions, per-user secrets
  actions.py         drafts & co. that need a human, with Gmail deep links
  google_oauth.py    per-user Google consent + token exchange
  session.py         the harness — turns, streaming, tools, confirmations, barge-in
  protocol.py        wire protocol types (mirrored by web/app.js)
  personality.py     persona loading/saving
  pipeline/          llm.py (Lunaroute), stt.py, tts.py, sentences.py
  tools/             base.py + registry.py + one module per integration
  integrations/      Google + Slack REST clients (fallback / OAuth plumbing)
  mcp/               MCP host: client.py, policy.py (tool mapping), manager.py
tts/
  server.py          local Qwen3-TTS streaming service (GPU 1)
web/                 the whole front end (no build step): app, login, dashboard
config/              app.yaml + personalities/
docs/PROTOCOL.md     exact wire protocol
scripts/             run_all.sh, run_tts.sh, google_auth.py, smoke_test.py
```

## Troubleshooting

**"replies in text only" in the UI** — the TTS service is not up. Check
`curl 127.0.0.1:8801/health` and `data/logs/tts.log`. First start downloads/loads the model
(~10–60 s).

**TTS crashes with `property 'upsample' … has no setter`** — `transformers` drifted above 5.15.
`pip install transformers==5.15.1` in `tts/.venv` (pinned in `tts/requirements.txt`).

**`CUDA initialization: driver too old`** — install the cu128 torch wheels
(`--index-url https://download.pytorch.org/whl/cu128`); Blackwell cards need them.

**“Microphone access was blocked”** — Chrome remembers a denied prompt per origin. Click the tune
icon in the address bar → *Microphone* → **Allow**, then reload and press the mic button. (Browsers
also need the click gesture before any `AudioContext` starts.)

**No audio in the browser** — browsers only allow an `AudioContext` after a gesture: press the mic
button once. Check the connection pill is green and the TTS badge is up.

**Gmail 403 / insufficient scope** — re-run `make google-auth` after enabling the APIs; the token
file is cached, so delete `secrets/google_token.json` first.

**Voice talks over me / cuts me off** — raise *end-of-speech* silence in settings, lower VAD
sensitivity, or use headphones (echo cancellation is on, but speakers at high volume still leak).

**Sentences still sound seamed** — check `OTTO_TTS_TRIM_SILENCE=1` (default) and that
`tts.min_chunk_chars` / `tts.coalesce_queue` are on in `config/app.yaml`; if you hear clicks
instead of gaps, the browser is not applying the gain envelope — reload with cache disabled
(`Page.reload ignoreCache`), since `app.js` is served with a plain ETag.

**Gmail/Calendar tools are missing or the Connections card says “Unavailable”** — the MCP services
are probably not enabled, or the OAuth client is not marked as an AI agent. See
[docs/MCP.md](docs/MCP.md); the card's *Check again* button and `POST /api/mcp/probe` show exactly
what Google answered.

**“Connect Google” says the OAuth client is not configured** — the server has no client yet: save
the JSON to `secrets/google_client.json` or set `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET`, and make
sure `OTTO_PUBLIC_URL + /oauth/google/callback` is registered as a redirect URI. See
[docs/AUTH.md](docs/AUTH.md).

**Signed out unexpectedly / everything 401s** — the session cookie was cleared or expired: go to
`/login`. Deleting `data/otto.db` resets every account and connection.

**The background-model toggle is greyed out** — the selected model has no `*-background` sibling on
Lunaroute. Pick another model or leave it off; jobs still run, just on the same model.

**Answers feel slow to start** — check the metrics strip (debug mode on) *before* blaming the
voice: `llm_ttft_ms` is usually the culprit, `first_audio_ms` rarely is. Try
`OTTO_STT_MODEL=whisper-large-v3`, lower `turn_taking.first_token_filler_ms`, or pick a different
model in the settings drawer.

**The UI says "speaking" while nothing plays** — the browser only reports `playback.idle` after
every scheduled buffer is finished; if it never arrives, the tab was backgrounded and Chrome
throttled the audio clock.

## Notes & limits

- Accounts are local and minimal: no email verification, no password reset. Signups are open to
  anyone who can reach the port, so keep it on `127.0.0.1` (or put an authenticating proxy in
  front) before exposing it.
- One conversation per websocket; history lives in memory (trimmed to `history_turns`).
- TTS runs one utterance at a time on one GPU; a second barge-in simply cancels the first.
- Transcription is utterance-level (parakeet is not a streaming model); live captions are
  best-effort re-transcriptions of the growing buffer.
