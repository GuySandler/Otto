# tts — local Qwen3-TTS streaming service

A small FastAPI wrapper around `faster-qwen3-tts` (Qwen3-TTS-12Hz-1.7B) that
gives the Otto server raw streaming PCM instead of a finished WAV. The Otto
server (`otto/pipeline/tts.py`) opens `POST /speak` and pipes the bytes straight
to the browser, so playback starts after the first ~200 ms instead of after the
whole sentence.

Pinned to one GPU (`OTTO_TTS_GPU`, default **1**) and one inference at a time: a
second request waits, and a client that hangs up (barge-in) makes the decoder
stop at the next chunk — that is the whole point of the streaming design.

## Run

```bash
./scripts/run_tts.sh                  # real model, GPU1, http://127.0.0.1:8801
OTTO_TTS_MOCK=1 ./scripts/run_tts.sh  # synthetic tone, no GPU needed (UI dev)
make tts                              # same as the first line
```

The venv is `tts/.venv` (`make setup-tts` recreates it). Always start the
service through the script, or export `CUDA_VISIBLE_DEVICES=1` yourself:
without it the model lands on GPU0 next to the desktop session.

## Environment

| var | default | meaning |
|---|---|---|
| `OTTO_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | HF id or local path; the name selects the mode (see below) |
| `OTTO_TTS_DEVICE` | `cuda:0` | torch device *inside* the process (GPU1 is device 0 once `CUDA_VISIBLE_DEVICES=1`) |
| `OTTO_TTS_GPU` | `1` | physical GPU, used by `scripts/run_tts.sh` as `CUDA_VISIBLE_DEVICES` |
| `OTTO_TTS_PORT` / `OTTO_TTS_HOST` | `8801` / `127.0.0.1` | bind address |
| `OTTO_TTS_CHUNK_SIZE` | `4` | codec steps per model chunk; 12 steps ≈ 1 s of audio, so 4 ≈ 0.35 s |
| `OTTO_TTS_INSTRUCT` | warm conversational instruction | style hint sent with every request (voice description in design mode) |
| `OTTO_TTS_MAX_CHARS` | `400` | hard clip, cut at a word boundary |
| `OTTO_TTS_MOCK` | `0` | `1` = fake backend for UI work |
| `OTTO_TTS_REF_AUDIO` / `OTTO_TTS_REF_TEXT` | – | reference clip + its transcript, required in clone mode |

Mode comes from the model id: `…-CustomVoice` → **custom** (preset speakers),
`…-VoiceDesign` → **design** (`instruct` describes the voice),
`…-Base` → **clone** (needs the two reference vars).

## Endpoints

| route | notes |
|---|---|
| `GET /health` | `ok`, `model`, `device`, `gpu`, `sample_rate`, `mode`, `ready`, `voices`, `uptime_s`, `loaded_in_s`, `error` |
| `GET /voices` | `{"default": "Aiden", "voices": [{id, description, language, kind}]}` |
| `POST /speak` | `{text, voice?, instruct?, language?="English", speed?=1.0}` → `audio/L16` stream, `X-Sample-Rate: 24000`, PCM16LE mono, chunked while generating. `400` empty text / unknown voice / unsupported language, `503` while the model is not ready |
| `GET /sample?voice=&text=` | complete WAV for the UI voice preview (`audio/wav`) |
| `POST /warmup` | `{"ok": true, "warmup_s": …}`; idempotent, single-flight |

`speed` is a soxr resample, i.e. tape-style: 1.5 is faster *and* higher.
faster-qwen3-tts offers no tempo control and pitch-exact stretching is not worth
a second dependency for a chat assistant.

Cancelling the HTTP request stops generation: the blocking generator runs in a
worker thread that checks a stop flag between chunks, and the single-flight lock
is only released after that thread exits, so a barge-in never overlaps the next
request on the GPU.

## Expected numbers

Measured on an RTX 5060 Ti 16 GB, `OTTO_TTS_CHUNK_SIZE=4`, warm model:

| metric | value |
|---|---|
| cold load (`/health.ready` after start) | ~9 s (weights + CUDA graph capture ~1 s) |
| TTFA, 50–100 char sentence | **~180 ms** (steady state; 250–330 ms while another process is using the GPU) |
| RTF (wall seconds per audio second) | **~0.45** (≈2.2× realtime, 0.43–0.79 observed) |
| VRAM | ~4.7 GB |

TTFA is dominated by chunk size: raising `OTTO_TTS_CHUNK_SIZE` costs first-audio
latency, lowering it costs throughput (more, smaller decodes).

## Troubleshooting

- **`property 'upsample' object has no setter` at load** — `transformers` moved
  off 5.15.1. 5.24 breaks Mimi encoder init; `tts/compat.py` (applied inside
  `QwenBackend.load`) only covers 5.15.x. Reinstall from `tts/requirements.txt`.
- **`no kernel image is available for execution on the device`** — the CPU/PyPI
  torch wheel is installed. Blackwell (RTX 50xx, sm_120) needs the CUDA 12.8
  wheels: `--extra-index-url https://download.pytorch.org/whl/cu128`.
- **`/health` says `ready: false`** — read the `error` field; the process keeps
  serving (`503` from `/speak`) instead of dying. Common causes: model id typo,
  clone mode without `OTTO_TTS_REF_AUDIO`/`OTTO_TTS_REF_TEXT`, OOM on a busy GPU.
- **CUDA out of memory** — another process is holding the card (a desktop
  session on GPU0 is why the service is pinned to GPU1). Check with
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.
- **`Address already in use`** — something else owns 8801 (`ss -ltnp | grep 8801`);
  the Otto server expects the TTS service exactly at `OTTO_TTS_URL`.
- **First request is slow, later ones are fast** — CUDA graph capture happens at
  load (or on the first generation). `POST /warmup` after start removes it.
