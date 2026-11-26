#!/usr/bin/env bash
# Run Otto's local Qwen3-TTS service (GPU1 by default, http://127.0.0.1:8801).
#
#   ./scripts/run_tts.sh                 # real model on CUDA device 1
#   OTTO_TTS_MOCK=1 ./scripts/run_tts.sh # fake 24 kHz backend, no GPU needed
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${OTTO_TTS_GPU:-1}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

HOST="${OTTO_TTS_HOST:-127.0.0.1}"
PORT="${OTTO_TTS_PORT:-8801}"
PYTHON="${OTTO_TTS_PYTHON:-tts/.venv/bin/python}"
MODEL="${OTTO_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
DEVICE="${OTTO_TTS_DEVICE:-cuda:0}"

echo "Otto TTS ───────────────────────────────────────────────────────────"
echo "  model  : ${MODEL}"
echo "  gpu    : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  (device ${DEVICE})"
echo "  url    : http://${HOST}:${PORT}"
if [[ "${OTTO_TTS_MOCK:-0}" == "1" ]]; then
  echo "  mode   : mock (synthetic tone, no model loaded)"
else
  echo "  mode   : from model name  chunk_size=${OTTO_TTS_CHUNK_SIZE:-4}"
fi
echo "────────────────────────────────────────────────────────────────────"

exec "${PYTHON}" tts/server.py --host "${HOST}" --port "${PORT}"
