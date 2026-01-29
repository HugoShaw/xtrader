from __future__ import annotations

from typing import Optional, List, Literal

from pydantic import BaseModel, Field, field_validator


class StockStrategyRequest(BaseModel):
    account_id: str = Field(..., min_length=1, max_length=64)
    symbol: str = Field(..., min_length=1, max_length=16)
    lookback_days: int = Field(default=5, ge=1, le=30)
    freq: Literal["1", "5", "15", "30", "60"] = "5"
    bar_type: Literal["minute", "daily"] = "daily"
    user_note: Optional[str] = Field(default=None, max_length=500)

    @field_validator("account_id")
    @classmethod
    def _norm_account_id(cls, v: str) -> str:
        vv = str(v or "").strip()
        if not vv:
            raise ValueError("account_id is required")
        return vv

    @field_validator("symbol")
    @classmethod
    def _norm_symbol(cls, v: str) -> str:
        vv = str(v or "").strip().upper()
        if not vv:
            raise ValueError("symbol is required")
        return vv


class StockStrategyAdvice(BaseModel):
    ok: bool = True
    account_id: str
    symbol: str
    data_window: str
    latest_close_cny: Optional[float] = None
    confidence_breakdown: dict = Field(default_factory=dict)
    summary: str
    action_bias: Literal["bullish", "bearish", "neutral"]
    plan: str
    key_levels: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    chart: List["ChartPoint"] = Field(default_factory=list)


class ChartPoint(BaseModel):
    ts: str
    close: float


class StockStrategyLLM(BaseModel):
    summary: str
    confidence_breakdown: dict = Field(default_factory=dict)
    action_bias: Literal["bullish", "bearish", "neutral"]
    plan: str
    key_levels: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
