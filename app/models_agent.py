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


class StockExpertRequest(BaseModel):
    account_id: str = Field(..., min_length=1, max_length=64)
    lookback_days: int = Field(default=540, ge=30, le=2000)
    adjust: str = Field(default="qfq", max_length=8)
    authorize_trade: bool = False
    execute: bool = False
    order_lots: int = Field(default=1, ge=1, le=20)
    max_symbols: int = Field(default=20, ge=1, le=50)

    @field_validator("account_id")
    @classmethod
    def _norm_account_id(cls, v: str) -> str:
        vv = str(v or "").strip()
        if not vv:
            raise ValueError("account_id is required")
        return vv


class ExpertTimeframeAdvice(BaseModel):
    timeframe: Literal["daily", "weekly", "monthly", "quarterly"]
    action_bias: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    last_close_cny: Optional[float] = None
    change_pct: Optional[float] = None
    short_ma: Optional[float] = None
    long_ma: Optional[float] = None


class ExpertNewsItem(BaseModel):
    title: str
    published_at: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    keywords: Optional[str] = None


class ExpertRealtimeQuote(BaseModel):
    ts: Optional[str] = None
    last_price_cny: Optional[float] = None
    amount: Optional[float] = None
    volume: Optional[float] = None


class StockExpertSymbolAdvice(BaseModel):
    symbol: str
    shares: int = 0
    cash_cny: float = 0.0
    strategy: List[ExpertTimeframeAdvice] = Field(default_factory=list)
    overall_action: Literal["BUY", "SELL", "HOLD"]
    overall_reason: str
    suggested_lots: Optional[int] = None
    news: List[ExpertNewsItem] = Field(default_factory=list)
    realtime: Optional[ExpertRealtimeQuote] = None
    execution: Optional[dict] = None


class StockExpertAdvice(BaseModel):
    ok: bool = True
    account_id: str
    asof: str
    authorized: bool = False
    executed: bool = False
    symbols: List[StockExpertSymbolAdvice] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class StockExpertLLM(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    suggested_lots: int = Field(ge=0, le=50)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    risk_notes: Optional[str] = None
