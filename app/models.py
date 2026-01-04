# app/models.py
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime

class Bar(BaseModel):
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

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
    suggested_notional_usd: float
    expected_direction: Literal["UP", "DOWN", "FLAT"]
    risk_notes: Optional[str] = None

class TradeRequest(BaseModel):
    symbol: str
    notional_usd: float
    action: Literal["BUY", "SELL"]

class TradeResult(BaseModel):
    ok: bool
    message: str
    ts: datetime
    symbol: str
    action: str
    notional_usd: float
    paper: bool = True
    details: Dict[str, Any] = Field(default_factory=dict)
