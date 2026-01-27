from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,32}$")


class SignupRequest(BaseModel):
    username: str = Field(..., description="3-32 chars: letters, digits, underscore")
    password: str = Field(..., min_length=8, max_length=128)
    password_confirm: str = Field(..., min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        vv = v.strip()
        if not USERNAME_RE.match(vv):
            raise ValueError("username must be 3-32 chars: letters, digits, underscore")
        return vv

    @model_validator(mode="after")
    def validate_passwords_match(self) -> "SignupRequest":
        if self.password != self.password_confirm:
            raise ValueError("password and password_confirm do not match")
        return self


class LoginRequest(BaseModel):
    username: str
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def normalize_username(cls, v: str) -> str:
        vv = v.strip()
        if not vv:
            raise ValueError("username is required")
        return vv


class UserPublic(BaseModel):
    id: int
    username: str
    created_at: Optional[str] = None
    last_login_at: Optional[str] = None


class AuthStatus(BaseModel):
    ok: bool = True
    user: Optional[UserPublic] = None
    detail: Optional[str] = None

