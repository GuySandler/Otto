#!/usr/bin/env python
"""One-time Google OAuth for Gmail + Calendar.

Usage: python scripts/google_auth.py

Opens a local browser flow against the OAuth *Desktop app* client stored at
``settings.google_client_secrets_path`` and writes the user token to
``settings.google_token_file`` (mode 600).
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: E402

from otto.integrations.google import SCOPES, build_credentials  # noqa: E402
from otto.settings import load_settings  # noqa: E402

USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

SETUP_HELP = """\
No OAuth client secrets found at {path}

Create an OAuth *Desktop app* client and save its JSON there:
  1. Open https://console.cloud.google.com/projectcreate and pick/create a project.
  2. Enable the Gmail API and the Google Calendar API for it.
  3. Go to https://console.cloud.google.com/apis/credentials -> Create credentials
     -> OAuth client ID -> Application type: *Desktop app*.
  4. Download the JSON and save it as {path}
     (or point GOOGLE_CLIENT_SECRETS in .env at it).
  5. Re-run: python scripts/google_auth.py
"""


def _id_token_claims(id_token: str) -> dict:
    """Decode (without verifying) the claims of the OIDC id_token."""
    try:
        payload = str(id_token).split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
    except (IndexError, ValueError):
        return {}
    return claims if isinstance(claims, dict) else {}


def _account_email(creds) -> str:
    claims = _id_token_claims(getattr(creds, "id_token", "") or "")
    email = str(claims.get("email") or "")
    if email:
        return email
    try:
        response = httpx.get(
            USERINFO_URL,
            headers={"Authorization": f"Bearer {creds.token}"},
            timeout=10.0,
        )
        response.raise_for_status()
        return str(response.json().get("email") or "")
    except Exception as exc:  # noqa: BLE001 - the email is informational only
        print(f"  (could not read the account email: {exc})")
        return ""


def main() -> int:
    settings = load_settings()
    secrets_path = settings.google_client_secrets_path
    if not secrets_path.exists():
        print(SETUP_HELP.format(path=secrets_path))
        return 1

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), SCOPES)
    creds = flow.run_local_server(port=0)

    token_path = settings.google_token_file
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    os.chmod(token_path, 0o600)

    # Read the credential back from disk so we verify exactly what the app will use.
    stored = build_credentials(settings)
    print(f"Saved Google token to {token_path}")
    print("Granted scopes:")
    for scope in stored.scopes or creds.scopes or []:
        print(f"  - {scope}")
    email = _account_email(stored)
    if email:
        print(f"Account: {email}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
