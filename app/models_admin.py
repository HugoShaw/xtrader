from __future__ import annotations

from typing import Optional, List

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models_auth import USERNAME_RE, UserPublic


class AdminUserCreate(BaseModel):
    username: str = Field(..., description="3-32 chars: letters, digits, underscore")
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        vv = v.strip()
        if not USERNAME_RE.match(vv):
            raise ValueError("username must be 3-32 chars: letters, digits, underscore")
        return vv


class AdminUserUpdate(BaseModel):
    username: Optional[str] = Field(default=None, description="3-32 chars: letters, digits, underscore")
    password: Optional[str] = Field(default=None, min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        vv = v.strip()
        if not USERNAME_RE.match(vv):
            raise ValueError("username must be 3-32 chars: letters, digits, underscore")
        return vv

    @model_validator(mode="after")
    def validate_any_update(self) -> "AdminUserUpdate":
        if not self.username and not self.password:
            raise ValueError("username or password is required")
        return self


class AdminUserList(BaseModel):
    items: List[UserPublic]
    total: int
    limit: int
    offset: int


class ApiUsageSummary(BaseModel):
    days: int
    series: list[dict]
    top_paths: list[dict]
    top_users: list[dict]
