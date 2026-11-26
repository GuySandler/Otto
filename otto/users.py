"""Users, sessions, and the per-user connection secrets (Google, Slack).

Local-first: one SQLite file, scrypt password hashes, opaque session tokens whose
SHA-256 is what actually lives in the database. Everything the tools need to act
*as the signed-in user* hangs off the user id.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .db import Database

log = logging.getLogger("otto.users")

SESSION_COOKIE = "otto_session"
SESSION_TTL_S = 60 * 60 * 24 * 30
MIN_PASSWORD = 8
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}


class AuthError(RuntimeError):
    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------- hashing
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)
    return "scrypt${n}${r}${p}${salt}${hash}".format(
        salt=base64.b64encode(salt).decode(),
        hash=base64.b64encode(digest).decode(),
        **_SCRYPT,
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=base64.b64decode(salt_b64),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(base64.b64decode(hash_b64)),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, base64.b64decode(hash_b64))


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def normalise_email(email: str) -> str:
    value = str(email or "").strip().lower()
    if not _EMAIL.match(value):
        raise AuthError("That email address doesn't look right.")
    return value


# ----------------------------------------------------------------------------- model
@dataclass(slots=True)
class User:
    id: str
    email: str
    name: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "email": self.email, "name": self.name, "created_at": self.created_at}

    @property
    def initials(self) -> str:
        source = self.name.strip() or self.email.split("@")[0]
        parts = [p for p in re.split(r"[.\s_-]+", source) if p]
        if len(parts) >= 2:
            return (parts[0][0] + parts[1][0]).upper()
        return source[:2].upper()


class UserStore:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._attempts: dict[str, list[float]] = {}

    # ------------------------------------------------------------------- users
    def count(self) -> int:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM users")
        return int(row["n"]) if row else 0

    def create_user(self, email: str, password: str, name: str = "") -> User:
        email = normalise_email(email)
        if len(password or "") < MIN_PASSWORD:
            raise AuthError(f"Password must be at least {MIN_PASSWORD} characters.")
        if self.db.query_one("SELECT id FROM users WHERE email = ?", (email,)):
            raise AuthError("An account with that email already exists.", status=409)
        user = User(id=uuid.uuid4().hex, email=email, name=str(name or "").strip()[:80], created_at=time.time())
        self.db.execute(
            "INSERT INTO users (id, email, name, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (user.id, user.email, user.name, hash_password(password), user.created_at),
        )
        log.info("user created: %s", user.email)
        return user

    def get(self, user_id: str) -> User | None:
        row = self.db.query_one("SELECT * FROM users WHERE id = ?", (user_id,))
        return self._row_to_user(row) if row else None

    def by_email(self, email: str) -> tuple[User, str] | None:
        row = self.db.query_one("SELECT * FROM users WHERE email = ?", (normalise_email(email),))
        if not row:
            return None
        return self._row_to_user(row), str(row["password_hash"])

    @staticmethod
    def _row_to_user(row: Any) -> User:
        return User(
            id=str(row["id"]),
            email=str(row["email"]),
            name=str(row["name"] or ""),
            created_at=float(row["created_at"]),
        )

    def delete_user(self, user_id: str, password: str) -> None:
        """Delete the account and everything hanging off it (secrets, tokens, sessions)."""
        found = self.db.query_one("SELECT password_hash FROM users WHERE id = ?", (user_id,))
        if found is None:
            raise AuthError("No such account.", status=404)
        if not verify_password(password, str(found["password_hash"])):
            raise AuthError("Wrong password.", status=401)
        self.db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        self.db.execute("DELETE FROM secrets WHERE user_id = ?", (user_id,))
        self.db.execute("DELETE FROM google_accounts WHERE user_id = ?", (user_id,))
        self.db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        log.info("user deleted: %s", user_id)

    # ------------------------------------------------------------------ sessions
    def login(self, email: str, password: str, *, user_agent: str = "") -> tuple[User, str]:
        self._rate_limit(str(email).lower())
        found = self.by_email(email)
        if found is None or not verify_password(password, found[1]):
            self._record_attempt(str(email).lower())
            raise AuthError("Wrong email or password.", status=401)
        user = found[0]
        token = self.new_session(user.id, user_agent=user_agent)
        return user, token

    def new_session(self, user_id: str, *, user_agent: str = "") -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        self.db.execute(
            "INSERT INTO sessions (token_hash, user_id, created_at, expires_at, user_agent) VALUES (?, ?, ?, ?, ?)",
            (_token_hash(token), user_id, now, now + SESSION_TTL_S, user_agent[:200]),
        )
        return token

    def user_for_token(self, token: str | None) -> User | None:
        if not token:
            return None
        row = self.db.query_one(
            "SELECT user_id, expires_at FROM sessions WHERE token_hash = ?", (_token_hash(token),)
        )
        if row is None:
            return None
        if float(row["expires_at"]) < time.time():
            self.db.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
            return None
        return self.get(str(row["user_id"]))

    def drop_session(self, token: str | None) -> None:
        if token:
            self.db.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))

    def purge_expired(self) -> None:
        self.db.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))

    # ------------------------------------------------------------ brute-force-ish
    def _rate_limit(self, email: str) -> None:
        cutoff = time.time() - 300
        recent = [t for t in self._attempts.get(email, []) if t > cutoff]
        self._attempts[email] = recent
        if len(recent) >= 8:
            raise AuthError("Too many attempts. Wait a couple of minutes.", status=429)

    def _record_attempt(self, email: str) -> None:
        self._attempts.setdefault(email, []).append(time.time())

    # ------------------------------------------------------------------- secrets
    def set_secret(self, user_id: str, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO secrets (user_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (user_id, key, value, time.time()),
        )

    def get_secret(self, user_id: str, key: str) -> str | None:
        row = self.db.query_one("SELECT value FROM secrets WHERE user_id = ? AND key = ?", (user_id, key))
        return str(row["value"]) if row else None

    def delete_secret(self, user_id: str, key: str) -> None:
        self.db.execute("DELETE FROM secrets WHERE user_id = ? AND key = ?", (user_id, key))

    # ------------------------------------------------------------ google account
    def save_google(self, user_id: str, token_json: str, *, email: str = "", scopes: list[str] | None = None) -> None:
        self.db.execute(
            "INSERT INTO google_accounts (user_id, token_json, email, scopes, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET token_json = excluded.token_json, email = excluded.email, "
            "scopes = excluded.scopes, updated_at = excluded.updated_at",
            (user_id, token_json, email, json.dumps(scopes or []), time.time()),
        )

    def get_google(self, user_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM google_accounts WHERE user_id = ?", (user_id,))
        if row is None:
            return None
        try:
            scopes = json.loads(row["scopes"] or "[]")
        except ValueError:
            scopes = []
        return {
            "token_json": str(row["token_json"]),
            "email": str(row["email"] or ""),
            "scopes": [str(s) for s in scopes],
            "updated_at": float(row["updated_at"]),
        }

    def delete_google(self, user_id: str) -> None:
        self.db.execute("DELETE FROM google_accounts WHERE user_id = ?", (user_id,))


# ------------------------------------------------------------------- token store
class UserGoogleTokenStore:
    """Adapter handed to :class:`otto.integrations.google.GoogleClient`.

    The client only knows load/save/clear, so the same class works with the
    per-user database or the legacy single-user JSON file.
    """

    def __init__(self, store: UserStore, user_id: str | None) -> None:
        self.store = store
        self.user_id = user_id
        self._cache: dict[str, Any] | None = None

    def load(self) -> str | None:
        if not self.user_id:
            return None
        record = self.store.get_google(self.user_id)
        self._cache = record
        return record["token_json"] if record else None

    def save(self, token_json: str) -> None:
        if not self.user_id:
            return
        email = (self._cache or {}).get("email") or ""
        scopes = (self._cache or {}).get("scopes") or []
        self.store.save_google(self.user_id, token_json, email=email, scopes=scopes)

    def clear(self) -> None:
        if self.user_id:
            self.store.delete_google(self.user_id)

    # ------------------------------------------------------------------ metadata
    @property
    def email(self) -> str:
        if self._cache is None:
            self.load()
        return str((self._cache or {}).get("email") or "")
