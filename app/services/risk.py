# app/services/risk.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, date
from app.models import TradeSignal

@dataclass
class RiskState:
    trades_today: int = 0
    last_trade_ts: datetime | None = None
    day: date = date.today()

    def reset_if_new_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            self.day = today
            self.trades_today = 0
            self.last_trade_ts = None

class RiskManager:
    def __init__(self, max_trades_per_day: int, cooldown_seconds: int, min_confidence: float):
        self.max_trades_per_day = max_trades_per_day
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence
        self.state = RiskState()

    def trades_left(self) -> int:
        self.state.reset_if_new_day()
        return max(0, self.max_trades_per_day - self.state.trades_today)

    def can_trade(self, signal: TradeSignal) -> tuple[bool, str]:
        self.state.reset_if_new_day()

        if self.state.trades_today >= self.max_trades_per_day:
            return False, "trade_limit_reached"

        if signal.confidence < self.min_confidence:
            return False, "confidence_too_low"

        if self.state.last_trade_ts is not None:
            dt = (datetime.now(timezone.utc) - self.state.last_trade_ts).total_seconds()
            if dt < self.cooldown_seconds:
                return False, "cooldown_active"

        if signal.action == "HOLD":
            return False, "hold_signal"

        return True, "ok"

    def record_trade(self):
        self.state.reset_if_new_day()
        self.state.trades_today += 1
        self.state.last_trade_ts = datetime.now(timezone.utc)
