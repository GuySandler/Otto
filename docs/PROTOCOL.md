# Otto wire protocol (browser ⇄ server)

Transport: one WebSocket at `/ws`. Text frames are JSON control messages; binary
frames carry raw PCM audio (both directions). The server also exposes a small
REST surface for bootstrap/config.

## Audio formats

| Direction | Format |
|---|---|
| browser → server | PCM16LE, mono, **16 000 Hz**, little-endian, contiguous mic audio |
| server → browser | PCM16LE, mono, sample rate given in the frame header (24000 Hz from Qwen3-TTS) |

Server → browser binary header (16 bytes, little-endian):

```
offset 0  : ASCII "AUD1"      (4 bytes)
offset 4  : uint32 seq        segment id (one per spoken sentence/chunk group)
offset 8  : uint32 sampleRate e.g. 24000
offset 12 : uint32 index      chunk index inside the segment (0,1,2,…)
offset 16 : PCM16LE payload
```

A segment is bracketed by `audio.start` / `audio.end` JSON messages with the same
`seq`. `audio.start` always precedes the first binary frame of that segment.
Segment ids are monotonically increasing but **not necessarily contiguous**: when
the server merges queued sentences into one synthesis request it keeps the
highest id of the merged range. Clients must therefore key off `audio.start`
and report the highest `seq` they played in `playback.idle`, not "the next one". `cancel.audio` with `fromSeq` tells the client to drop every buffered
segment with `seq >= fromSeq` **immediately** (barge-in).

## Client → server (JSON)

| type | fields | meaning |
|---|---|---|
| `hello` | `sampleRate?` | first message after open; server replies `ready` |
| `utterance.start` | `ts` | VAD says the user started talking |
| `utterance.end` | `ts`, `durationMs` | VAD says the utterance ended; server transcribes the buffered audio |
| `barge_in` | `ts?` | user talked over the assistant (client has already stopped local playback) |
| `text` | `text` | typed input (bypasses STT) |
| `confirm` | `id`, `approved` | button answer to a `confirm.request` |
| `cancel` | – | stop speaking / abort the current turn, keep listening |
| `playback.idle` | `seq` | client finished playing every segment up to `seq` |
| `config` | `model?`, `voice?`, `instruct?`, `personality?`, `speed?`, `stt_model?`, `parallel_agents?`, `background_model?` | update session config |
| `job.cancel` | `id` | cancel a running background job |
| `ping` | – | keepalive; server answers `pong` |

## Server → client (JSON)

| type | fields | meaning |
|---|---|---|
| `ready` | `session`, `config`, `personality`, `state`, `vad` | session bootstrap (voices/models come from `/api/bootstrap`) |
| `state` | `state`: `idle`\|`listening`\|`thinking`\|`speaking`\|`awaiting_confirmation`, `detail?` | UI state |
| `stt.partial` | `text` | live transcript while the user speaks (best effort) |
| `stt.final` | `text`, `durationMs`, `latencyMs` | final transcript of an utterance |
| `user.text` | `text`, `source` | accepted user message (`voice`\|`typed`) |
| `assistant.delta` | `id`, `text` | streaming model text token(s) for display |
| `assistant.message` | `id`, `text`, `spoken` | completed assistant message |
| `assistant.thinking` | `id`, `text?` | model is reasoning (optional UI hint) |
| `audio.start` | `seq`, `sampleRate`, `text` | a TTS segment is starting |
| `audio.end` | `seq`, `ttfaMs?`, `ms?` | segment finished |
| `cancel.audio` | `fromSeq` | flush playback |
| `tool.call` | `id`, `name`, `args`, `category?` | tool about to run |
| `tool.result` | `id`, `name`, `ok`, `summary`, `data?`, `display?`, `ms` | tool finished |
| `confirm.request` | `id`, `tool`, `name`, `args`, `summary`, `preview?`, `danger` | needs a yes/no |
| `confirm.resolved` | `id`, `approved` | confirmation answered |
| `notice` | `level`, `message` | non-fatal info (degraded TTS, auth missing, …) |

Tool `display` kinds produced by the built-in tools: `email_list`, `email`,
`email_draft`, `email_sent`, `event_list`, `event_created`, `event_updated`,
`event_deleted`, `slack_messages`, `slack_sent`, `slack_channels`,
`subagents`, `job_started`, `jobs`, `list`, `text`.
| `error` | `message` | fatal-ish error for the turn |
| `job.update` | `id`, `title`, `kind`, `status`, `ms`, `result?`, `error?` | background job lifecycle (`queued`/`running`/`done`/`failed`/`cancelled`) |
| `job.cancelled` | `id`, `ok` | answer to `job.cancel` |
| `action.created` | `action`, `open` | a tool produced something the user must finish (e.g. an email draft); `open` is the new count for the dashboard badge |
| `metrics` | `latency` | per-turn timings (`stt_ms`, `stt_model`, `llm_ttft_ms`, `first_audio_ms`, `speech_start_ms`, `turn_ms`) |
| `pong` | – | keepalive |

## REST

| route | meaning |
|---|---|
| `GET /api/health` | server + lunaroute + tts health |
| `GET /api/bootstrap` | models, `sttModels`, personalities, voices, tool catalog, current config, `capabilities` (`parallel_agents`, `background_model{available,id}`, `jobs`) |
| `POST /api/config` | persist default session config (model, voice, personality, `stt_model`, `parallel_agents`, `background_model`) |
| `GET /api/personalities` · `GET/PUT /api/personalities/{id}` | read/write personality prompts |
| `GET /api/tools` | tool catalog (name, description, schema, category, requires_confirmation) |
| `GET /api/actions?status=open\|done\|dismissed\|all` | things waiting on the user (drafts, …) with deep links |
| `POST /api/actions/{id}` `{"status":"done"\|"dismissed"\|"open"}` | finish / dismiss / reopen one |
| `POST /api/mcp/probe` | re-run MCP discovery and return the raw per-server state |
| `GET /api/voice-sample?voice=` | WAV preview of a TTS voice |
| `POST /api/warmup` | warm the local TTS model |

## Turn-taking / barge-in semantics

1. Client VAD detects speech onset. If the assistant is speaking, the client
   **stops playback instantly on its own** and immediately sends `barge_in`.
2. Server cancels the in-flight LLM stream and TTS jobs, emits `cancel.audio`
   (`fromSeq` = the first segment id of the cancelled turn) and returns to `listening`.
3. Client streams mic audio; on `utterance.end` the server transcribes and runs
   a turn.
4. If a confirmation is pending, the next `utterance.end` is interpreted as an
   answer (approve / deny / off-topic).

Turn end is not "assistant finished generating" but "assistant finished
**speaking**": the server stays in `speaking` while the model is still producing
text, and only goes `idle` once the model is done *and* the client has reported
`playback.idle` for the highest audio `seq` it was sent (so a pause between
sentences never flickers the UI).
