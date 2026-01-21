# app/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, field_serializer

from app.utils.timeutils import ensure_shanghai, fmt_shanghai


class Bar(BaseModel):
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    # from ak.stock_zh_a_hist_min_em()
    amount: Optional[float] = None   # 成交额
    vwap: Optional[float] = None     # 均价 (avg price)

    @field_validator("ts", mode="before")
    @classmethod
    def _ts_shanghai(cls, v: Any) -> datetime:
        # Pydantic may pass datetime or string here; let it parse first if needed.
        # If it's already datetime, ensure_shanghai will fix tz.
        if isinstance(v, datetime):
            return ensure_shanghai(v)  # type: ignore[return-value]
        return v  # let Pydantic parse, then we'll ensure in a second validator

    @field_validator("ts")
    @classmethod
    def _ts_shanghai_after(cls, v: datetime) -> datetime:
        return ensure_shanghai(v)  # type: ignore[return-value]

    @field_serializer("ts")
    def _ser_ts(self, v: datetime) -> str:
        # Output as "YYYY-MM-DD HH:MM:SS" in Shanghai (no offset)
        return fmt_shanghai(v) or ensure_shanghai(v).isoformat()


class MarketSnapshot(BaseModel):
    symbol: str
    ts: datetime  # snapshot time in Asia/Shanghai (tz-aware)
    last_price: float
    day_open: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    day_volume: Optional[float] = None
    vwap: Optional[float] = None
    prev_close: Optional[float] = None
    recent_bars: List[Bar] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("ts", mode="before")
    @classmethod
    def _ts_shanghai(cls, v: Any) -> datetime:
        if isinstance(v, datetime):
            return ensure_shanghai(v)  # type: ignore[return-value]
        return v

    @field_validator("ts")
    @classmethod
    def _ts_shanghai_after(cls, v: datetime) -> datetime:
        return ensure_shanghai(v)  # type: ignore[return-value]

    @field_serializer("ts")
    def _ser_ts(self, v: datetime) -> str:
        return fmt_shanghai(v) or ensure_shanghai(v).isoformat()


class TradeSignal(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    horizon_minutes: int = 30
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    suggested_lots: int = Field(ge=0, le=1000)
    expected_direction: Literal["UP", "DOWN", "FLAT"]
    risk_notes: str = ""


class TradeRequest(BaseModel):
    symbol: str
    action: Literal["BUY", "SELL"]
    shares: int = Field(gt=0)          # final order size in shares
    lot_size: int = Field(gt=0)        # keep for audit/debug

    # NEW: optional fill reference price (strategy should pass snapshot.last_price)
    price_cny: Optional[float] = Field(default=None, gt=0)

# app/models.py
class TradeResult(BaseModel):
    ok: bool
    message: str
    ts: datetime  # in Asia/Shanghai (tz-aware)
    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    shares: int = 0
    lots: int = 0
    paper: bool = True

    # NEW (all optional for backward compatibility)
    executed_price_cny: Optional[float] = None        # raw fill/last price
    executed_price_net_cny: Optional[float] = None    # effective price after cost
    notional_cny: Optional[float] = None
    fee_cny: Optional[float] = None
    slippage_cny: Optional[float] = None
    total_cost_cny: Optional[float] = None
    cash_delta_cny: Optional[float] = None            # cash impact (BUY negative, SELL positive)

    details: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("ts", mode="before")
    @classmethod
    def _ts_shanghai(cls, v: Any) -> datetime:
        if isinstance(v, datetime):
            return ensure_shanghai(v)  # type: ignore[return-value]
        return v

    @field_validator("ts")
    @classmethod
    def _ts_shanghai_after(cls, v: datetime) -> datetime:
        return ensure_shanghai(v)  # type: ignore[return-value]

    @field_serializer("ts")
    def _ser_ts(self, v: datetime) -> str:
        return fmt_shanghai(v) or ensure_shanghai(v).isoformat()


@dataclass(frozen=True)
class AccountState:
    cash_cny: float
    position_shares: int
    avg_cost_cny: Optional[float] = None
    unrealized_pnl_cny: Optional[float] = None


@dataclass(frozen=True)
class ExecutionConstraints:
    horizon_minutes: int = 30
    max_trades_left_today: int = 0
    lot_size: int = 100
    max_order_shares: Optional[int] = None
    min_confidence_to_trade: float = 0.65
    fees_bps_est: int = 5
    slippage_bps_est: int = 5
    currency: str = "CNY"

    # NEW: contract / market data availability hints (provider-specific but normalized)
    market_contract: Dict[str, Any] = field(default_factory=dict)

# -------------------------
# Request models for /signal and /execute
# -------------------------
class AccountStateIn(BaseModel):
    cash_cny: float = Field(..., ge=0.0)
    position_shares: int = Field(..., ge=0)
    avg_cost_cny: Optional[float] = None
    unrealized_pnl_cny: Optional[float] = None


class MarketOptionsIn(BaseModel):
    """
    Per-call market provider options.

    Keep it conservative: only expose a few knobs you actually support, but
    still allow additional keys for provider-specific experimentation.
    """
    min_period: Optional[str] = Field(default="1", description="Provider min period (string).")
    min_lookback_minutes: Optional[int] = Field(default=120, ge=1, description="Lookback minutes for intraday bars.")
    include_orderbook: bool = Field(default=False, description="Whether to include orderbook (expensive).")
    include_bars: bool = Field(default=True, description="Whether to include bars.")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific extra kwargs.")

    def to_kwargs(self) -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            "min_period": self.min_period,
            "min_lookback_minutes": self.min_lookback_minutes,
            "include_orderbook": self.include_orderbook,
            "include_bars": self.include_bars,
        }
        # merge extra last (extra can override)
        kw.update(self.extra or {})
        # drop None keys to avoid confusing providers
        return {k: v for k, v in kw.items() if v is not None}


class ExecutionOptionsIn(BaseModel):
    """
    Per-call StrategyEngine overrides.

    Notes:
      - max_order_shares:
          None => no cap
          0    => interpreted as no cap as well (client-friendly)
    """
    timezone_name: str = Field(default="Asia/Shanghai")
    lot_size: int = Field(default=100, ge=1)
    max_order_shares: Optional[int] = Field(default=1000, ge=0)
    fees_bps_est: int = Field(default=5, ge=0, le=10_000)
    slippage_bps_est: int = Field(default=5, ge=0, le=10_000)
    market: MarketOptionsIn = Field(default_factory=MarketOptionsIn)

    def normalized_max_order_shares(self) -> Optional[int]:
        if self.max_order_shares is None:
            return None
        if int(self.max_order_shares) == 0:
            return None
        return int(self.max_order_shares)


class TradingContextIn(BaseModel):
    """
    StrategyEngine requires account state.
    Allow overriding "now_ts" for reproducible testing.

    now_ts format: 'YYYY-MM-DD HH:MM:SS' in Asia/Shanghai (no offset).
    """
    account_state: AccountStateIn
    now_ts: Optional[str] = Field(
        default=None,
        description='Optional decision timestamp string like "YYYY-MM-DD HH:MM:SS" in Asia/Shanghai.',
    )

    # NEW: optional per-call overrides for strategy
    options: ExecutionOptionsIn = Field(default_factory=ExecutionOptionsIn)