from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from fastapi import HTTPException, Request, Response, status

from app.config import settings
from app.logging_config import logger
from app.models_auth import UserPublic
from app.storage.auth_repo import UserRepo
from app.storage.db import session_scope


PBKDF2_ITERATIONS = 200_000


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + pad).encode("ascii"))


def _pbkdf2_hash(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return dk.hex()


def hash_password(password: str) -> Tuple[str, str]:
    salt = secrets.token_urlsafe(16)
    return _pbkdf2_hash(password, salt), salt


def verify_password(password: str, *, salt: str, password_hash: str) -> bool:
    calc = _pbkdf2_hash(password, salt)
    return hmac.compare_digest(calc, password_hash)


def _sign(data: bytes) -> str:
    secret = settings.auth_secret_key.encode("utf-8")
    sig = hmac.new(secret, data, hashlib.sha256).digest()
    return _b64url_encode(sig)


def _encode_session(username: str, *, ttl_seconds: int) -> str:
    exp = int(time.time()) + int(ttl_seconds)
    payload = {"u": username, "exp": exp}
    payload_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload_b64 = _b64url_encode(payload_bytes)
    sig_b64 = _sign(payload_b64.encode("utf-8"))
    return f"{payload_b64}.{sig_b64}"


def _decode_session(token: str) -> Optional[dict]:
    try:
        payload_b64, sig_b64 = token.split(".", 1)
    except ValueError:
        return None

    expected_sig = _sign(payload_b64.encode("utf-8"))
    if not hmac.compare_digest(expected_sig, sig_b64):
        return None

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:
        return None

    exp = int(payload.get("exp") or 0)
    if exp <= int(time.time()):
        return None

    return payload


def get_session_username(token: str) -> Optional[str]:
    payload = _decode_session(token)
    if not payload:
        return None
    username = str(payload.get("u") or "").strip()
    return username or None


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        max_age=int(settings.auth_session_ttl_seconds),
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(settings.auth_cookie_name, path="/")


@dataclass
class AuthenticatedUser:
    id: int
    username: str
    created_at: Optional[str] = None
    last_login_at: Optional[str] = None

    def to_public(self) -> UserPublic:
        return UserPublic(
            id=self.id,
            username=self.username,
            created_at=self.created_at,
            last_login_at=self.last_login_at,
        )


class AuthService:
    def __init__(self, request: Request):
        self.request = request
        self.session_factory = request.app.state.session_factory

    async def signup(self, *, username: str, password: str) -> AuthenticatedUser:
        pwd_hash, salt = hash_password(password)

        async with session_scope(self.session_factory) as s:
            repo = UserRepo(s)
            existing = await repo.get_by_username(username)
            if existing:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="username already exists")

            user = await repo.create_user(username=username, password_hash=pwd_hash, salt=salt)
            logger.info("auth_signup | username=%s user_id=%s", username, user.id)
            return AuthenticatedUser(id=user.id, username=user.username)

    async def login(self, *, username: str, password: str) -> Tuple[AuthenticatedUser, str]:
        async with session_scope(self.session_factory) as s:
            repo = UserRepo(s)
            user = await repo.get_by_username(username)
            if not user or not verify_password(password, salt=user.salt, password_hash=user.password_hash):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid username or password")

            await repo.update_last_login(user)

            token = _encode_session(username=user.username, ttl_seconds=settings.auth_session_ttl_seconds)
            logger.info("auth_login | username=%s user_id=%s", username, user.id)

            return (
                AuthenticatedUser(
                    id=user.id,
                    username=user.username,
                    created_at=user.created_at.isoformat() if user.created_at else None,
                    last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
                ),
                token,
            )

    async def current_user(self) -> AuthenticatedUser:
        token = self.request.cookies.get(settings.auth_cookie_name)
        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="not authenticated")

        payload = _decode_session(token)
        if not payload:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or expired session")

        username = str(payload.get("u") or "")
        if not username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid session payload")

        async with session_scope(self.session_factory) as s:
            repo = UserRepo(s)
            user = await repo.get_by_username(username)
            if not user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="user not found")

            return AuthenticatedUser(
                id=user.id,
                username=user.username,
                created_at=user.created_at.isoformat() if user.created_at else None,
                last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
            )


def is_superuser(user: AuthenticatedUser) -> bool:
    raw = settings.admin_usernames or ""
    admins = {u.strip() for u in raw.split(",") if u.strip()}
    return user.username in admins


def get_auth_service(request: Request) -> AuthService:
    return AuthService(request)
