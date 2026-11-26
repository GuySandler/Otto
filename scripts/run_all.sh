#!/usr/bin/env bash
# Run Otto end-to-end: local Qwen3-TTS (GPU) + the harness server.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a; . ./.env; set +a
fi

if [[ -z "${LUNAROUTE_API_KEY:-}" ]]; then
  echo "!! LUNAROUTE_API_KEY is not set — transcription and the model will fail."
  echo "   Copy .env.example to .env and add your key."
fi

mkdir -p data/logs
TTS_LOG="data/logs/tts.log"

cleanup() {
  if [[ -n "${TTS_PID:-}" ]] && kill -0 "$TTS_PID" 2>/dev/null; then
    echo "→ stopping tts (pid $TTS_PID)"
    kill "$TTS_PID" 2>/dev/null || true
    wait "$TTS_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "→ starting Qwen3-TTS on GPU ${OTTO_TTS_GPU:-1} (log: $TTS_LOG)"
OTTO_TTS_GPU="${OTTO_TTS_GPU:-1}" ./scripts/run_tts.sh >"$TTS_LOG" 2>&1 &
TTS_PID=$!

# wait for the voice to come up (model load can take a while on first run)
if command -v curl >/dev/null; then
  for _ in $(seq 1 120); do
    if curl -fsS "${OTTO_TTS_URL:-http://127.0.0.1:8801}/health" >/dev/null 2>&1; then
      echo "→ tts ready"
      break
    fi
    sleep 2
  done
fi

echo "→ starting Otto on http://${OTTO_HOST:-127.0.0.1}:${OTTO_PORT:-8800}"
exec ./.venv/bin/python -m uvicorn otto.main:app \
  --host "${OTTO_HOST:-127.0.0.1}" --port "${OTTO_PORT:-8800}" \
  --log-level "${OTTO_LOG_LEVEL:-info}"
