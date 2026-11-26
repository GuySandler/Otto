.PHONY: help setup setup-tts tts server dev test clean google-auth smoke

help:
	@echo "Otto — voice assistant harness"
	@echo ""
	@echo "  make setup        install python deps (app + dev) and the GPU TTS env"
	@echo "  make tts          run the local Qwen3-TTS service (GPU1, port 8801)"
	@echo "  make server       run the Otto server (port 8800, reload)"
	@echo "  make dev          run the TTS service and the server together"
	@echo "  make test         run the test suite"
	@echo "  make google-auth  one-time Google OAuth for Gmail + Calendar"
	@echo "  make smoke        end-to-end headless smoke test (needs API key)"

PY := .venv/bin/python
UV := uv

setup: setup-tts
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python .venv/bin/python -e ".[dev]"
	@echo "Copy .env.example to .env and add LUNAROUTE_API_KEY"

setup-tts:
	$(UV) venv --python 3.12 tts/.venv
	$(UV) pip install --python tts/.venv/bin/python -r tts/requirements.txt

tts:
	./scripts/run_tts.sh

server:
	$(PY) -m uvicorn otto.main:app --host 127.0.0.1 --port 8800 --reload

dev:
	./scripts/run_all.sh

test:
	$(PY) -m pytest -q

google-auth:
	$(PY) scripts/google_auth.py

smoke:
	$(PY) scripts/smoke_test.py

clean:
	rm -rf .pytest_cache **/__pycache__
