# Accounts, sessions and tool connections

Otto is a local-first app, but it is no longer a single anonymous user: there are
accounts, a login page, and — most importantly — **each account has its own
Gmail/Calendar (and optional Slack) credentials**. Two people on the same server
never see each other's mail.

```
browser ── cookie: otto_session ──▶ FastAPI  ──▶ users table (SQLite)
                                     │
                                     └─▶ per-user Google refresh token ─▶ Gmail / Calendar
                                     └─▶ per-user Slack bot token     ─▶ Slack
```

## Accounts

* First visit sends you to `/login`. Create an account with an email + password
  (8+ characters). There is no email verification and no password reset yet —
  this is a single-machine tool, not a SaaS.
* Passwords are stored as `scrypt` hashes (n=2¹⁴, per-user random salt).
* Sessions are opaque 32-byte tokens; only their SHA-256 lives in the database.
  The cookie is `HttpOnly`, `SameSite=Lax`, `Secure` when served over HTTPS, and
  expires after 30 days. `POST /api/auth/logout` deletes the row server-side.
* Wrong-password attempts are rate limited (8 per 5 minutes per email).
* Everything lives in `data/otto.db` (SQLite, WAL). Back it up to back up the
  accounts; delete it to start over.

| route | who | what |
|---|---|---|
| `GET /login` | anyone | login / signup page |
| `POST /api/auth/signup` | anyone | `{email, password, name?}` |
| `POST /api/auth/login` | anyone | `{email, password}` |
| `POST /api/auth/logout` | signed in | drops the session |
| `GET /api/auth/me` | signed in | the current user |
| everything else | signed in | `/api/*` returns 401, `/` redirects to `/login`, the websocket closes with `4401` |

Mutating requests are also rejected when the `Origin` header does not match
`Host`, which — combined with `SameSite=Lax` and JSON-only bodies — is enough to
stop drive-by cross-site writes on a localhost app.

## Connecting Google (Gmail + Calendar)

This is the "log in with Google" flow. It needs **one** OAuth client for the
server; every user then consents with their own Google account.

1. <https://console.cloud.google.com> → new project (e.g. `otto`).
2. **APIs & Services → Library** → enable **Gmail API** and **Google Calendar API**.
3. **OAuth consent screen**: External · Testing is fine → add your own Google
   address under *Test users* (Google will refuse other accounts until the app is
   published, which you do not need locally).
4. **Credentials → Create credentials → OAuth client ID → *Web application*** (not
   “Desktop app” — that type is only for `scripts/google_auth.py`, which has no
   redirect path):
   * Authorized redirect URI — must match exactly:
     ```
     http://127.0.0.1:8800/oauth/google/callback
     ```
     (if you changed `OTTO_HOST`/`OTTO_PORT`, or serve behind a proxy, set
     `OTTO_PUBLIC_URL` and register `<that URL>/oauth/google/callback`)
   * Download the JSON and save it as `secrets/google_client.json` — or just drop
     the downloaded `client_secret_….json` anywhere in the repo root or `secrets/`;
     Otto finds it (the file is read on every request, so no restart needed).
   * Or skip the file and put `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` in `.env`.
5. Restart Otto, open **Settings → Connections → Connect Google**, approve, and you
   come back to `/?connected=google`. The account email appears on the card.
6. The Gmail and Calendar tools now appear for *that* user only.

Scopes requested: `gmail.modify`, `gmail.send`, `calendar.events`,
`userinfo.email`. Only the scopes the tools actually need — nothing else.

**Disconnect** (Settings → Connections → Disconnect) deletes the stored token; the
tools disappear from that user's toolbelt immediately. Revoking access at
<https://myaccount.google.com/permissions> has the same effect on the next call.

### If something goes wrong

| symptom | cause |
|---|---|
| `?error=google_exchange_failed` | `redirect_uri_mismatch` — the URI in Google Cloud must match `OTTO_PUBLIC_URL + /oauth/google/callback` byte for byte (http vs https, 127.0.0.1 vs localhost, trailing slash). |
| `Access blocked: app has not completed verification` | add your account under *Test users* on the OAuth consent screen. |
| `?error=google_bad_state` | you took more than 10 minutes in the consent screen, or the flow was started in another tab/session. Click Connect again. |
| `?error=google_no_code` | you pressed Cancel/Deny. |
| Tool says “Google is not connected” mid-conversation | the refresh token was revoked or the password changed. Settings → Connections → Connect again. |
| `configured: false` on the card | no OAuth client on the server yet — step 4 above. |

### CLI fallback (single-user, no accounts)

`python scripts/google_auth.py` still works: it writes
`secrets/google_token.json` and the tools use it when no user session exists
(scripts, the smoke test, dev machines). The web app never reads that file for a
signed-in user — it is strictly per-account there.

## Connecting Slack

Slack does not have a public "log in with Slack" for bots that is worth the
ceremony here, so the UI takes a bot token:

1. <https://api.slack.com/apps> → *Create New App* → *From scratch*.
2. **OAuth & Permissions → Bot Token Scopes**: `channels:history`,
   `channels:read`, `groups:history`, `chat:write`, `im:history`, `search:read`,
   `users:read`, `users:read.email`.
3. *Install to Workspace* → copy the `xoxb-…` **Bot User OAuth Token**.
4. Invite the bot to the channels it should read (`/invite @otto`).
5. Paste the token in **Settings → Connections → Slack → Save**. Otto validates it
   with one cheap API call and refuses it if Slack disagrees.

Tokens are write-only in the UI, stored per user in `data/otto.db`, and deleted by
*Remove*. `SLACK_BOT_TOKEN` in `.env` still works as a server-wide fallback.

## Where credentials live

| what | where | rotated by |
|---|---|---|
| password | `data/otto.db → users.password_hash` (scrypt) | user (no reset flow yet) |
| session | `data/otto.db → sessions` (sha256 of the token) | logout / 30 days |
| Google refresh token | `data/otto.db → google_accounts` | reconnect |
| Slack bot token | `data/otto.db → secrets` | reconnect |

No credential is ever written to logs, and nothing is echoed back to the browser —
`/api/connections` only says *connected / not connected / which email*.

## Security notes for this build

* Binds `127.0.0.1` by default. If you expose it, terminate TLS in front and set
  `OTTO_PUBLIC_URL=https://…` so the session cookie gets the `Secure` flag.
* Signups are open to anyone who can reach the server. There is no invite code —
  do not put this on the public internet without one.
* One conversation per websocket, conversation history in memory; a background job
  finishes even if you close the tab (it lives in the server process).
