# app/models.py
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime

class Bar(BaseModel):
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    # NEW: from ak.stock_zh_a_hist_min_em()
    amount: Optional[float] = None   # 成交额
    vwap: Optional[float] = None     # 均价 (avg price)

class MarketSnapshot(BaseModel):
    symbol: str
    ts: datetime  # snapshot time (UTC recommended)
    last_price: float
    day_open: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    day_volume: Optional[float] = None
    vwap: Optional[float] = None
    prev_close: Optional[float] = None
    recent_bars: List[Bar] = Field(default_factory=list)  # e.g., last N 1-min bars
    extra: Dict[str, Any] = Field(default_factory=dict)

class TradeSignal(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    horizon_minutes: int = 30
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    suggested_notional_cny: float
    expected_direction: Literal["UP", "DOWN", "FLAT"]
    risk_notes: str = ""   # <-- make required + safe default

class TradeRequest(BaseModel):
    symbol: str
    notional_cny: float
    action: Literal["BUY", "SELL"]

class TradeResult(BaseModel):
    ok: bool
    message: str
    ts: datetime
    symbol: str
    action: str
    notional_cny: float
    paper: bool = True
    details: Dict[str, Any] = Field(default_factory=dict)

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

# -------------------------
# Request models for /signal and /execute
# -------------------------
class AccountStateIn(BaseModel):
    cash_cny: float = Field(..., ge=0.0)
    position_shares: int = Field(..., ge=0)
    avg_cost_cny: Optional[float] = None
    unrealized_pnl_cny: Optional[float] = None


class TradingContextIn(BaseModel):
    """
    Your StrategyEngine now requires account state.
    We also allow overriding "now_ts" for reproducible testing.
    """
    account_state: AccountStateIn
    now_ts: Optional[str] = Field(
        default=None,
        description='Optional decision timestamp string like "YYYY-MM-DD HH:MM:SS". If omitted, snapshot.ts is used.',
    )
