from __future__ import annotations

from typing import Optional, List

from pydantic import BaseModel, Field, field_validator, model_validator


class TradeAccountBase(BaseModel):
    account_id: str = Field(..., min_length=1, max_length=64)
    name: Optional[str] = Field(default=None, max_length=128)
    cash_cny: float = Field(default=0.0, ge=0.0)
    base_currency: str = Field(default="CNY", min_length=1, max_length=8)

    @field_validator("account_id")
    @classmethod
    def _normalize_account_id(cls, v: str) -> str:
        vv = str(v or "").strip()
        if not vv:
            raise ValueError("account_id is required")
        return vv

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        vv = v.strip()
        return vv or None

    @field_validator("base_currency")
    @classmethod
    def _normalize_currency(cls, v: str) -> str:
        vv = str(v or "").strip().upper()
        if not vv:
            raise ValueError("base_currency is required")
        return vv


class TradeAccountCreate(TradeAccountBase):
    positions: List["TradePositionCreate"] = Field(default_factory=list)


class TradeAccountUpdate(BaseModel):
    name: Optional[str] = Field(default=None, max_length=128)
    cash_cny: Optional[float] = Field(default=None, ge=0.0)
    base_currency: Optional[str] = Field(default=None, min_length=1, max_length=8)

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        vv = v.strip()
        return vv or None

    @field_validator("base_currency")
    @classmethod
    def _normalize_currency(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        vv = str(v or "").strip().upper()
        if not vv:
            raise ValueError("base_currency is required")
        return vv

    @model_validator(mode="after")
    def _validate_any_update(self) -> "TradeAccountUpdate":
        if self.name is None and self.cash_cny is None and self.base_currency is None:
            raise ValueError("at least one field is required")
        return self


class TradePositionBase(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=16)
    shares: int = Field(default=0, ge=0)
    avg_cost_cny: Optional[float] = Field(default=None, ge=0.0)
    unrealized_pnl_cny: Optional[float] = Field(default=None)

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, v: str) -> str:
        vv = str(v or "").strip().upper()
        if not vv:
            raise ValueError("symbol is required")
        return vv


class TradePositionCreate(TradePositionBase):
    pass


class TradePositionUpdate(BaseModel):
    shares: Optional[int] = Field(default=None, ge=0)
    avg_cost_cny: Optional[float] = Field(default=None, ge=0.0)
    unrealized_pnl_cny: Optional[float] = Field(default=None)

    @model_validator(mode="after")
    def _validate_any_update(self) -> "TradePositionUpdate":
        if self.shares is None and self.avg_cost_cny is None and self.unrealized_pnl_cny is None:
            raise ValueError("at least one field is required")
        return self


class TradePositionOut(TradePositionBase):
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TradeAccountOut(TradeAccountBase):
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TradeAccountDetail(TradeAccountOut):
    positions: List[TradePositionOut] = Field(default_factory=list)


class TradeAccountList(BaseModel):
    items: List[TradeAccountOut]
    total: int
