# app/services/risk.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from app.models import TradeSignal, AccountState, MarketSnapshot

CN_TZ = ZoneInfo("Asia/Shanghai")


@dataclass
class RiskState:
    trades_today: int = 0
    last_trade_ts: datetime | None = None
    day: datetime.date = datetime.now(CN_TZ).date()

    def reset_if_new_day(self):
        today = datetime.now(CN_TZ).date()
        if today != self.day:
            self.day = today
            self.trades_today = 0
            self.last_trade_ts = None


class RiskManager:
    def __init__(
        self,
        *,
        max_trades_per_day: int,
        cooldown_seconds: int,
        min_confidence: float,
        max_position_value_cny: float,
        lot_size: int = 100,
    ):
        self.max_trades_per_day = int(max_trades_per_day)
        self.cooldown_seconds = int(cooldown_seconds)
        self.min_confidence = float(min_confidence)
        self.max_position_value_cny = float(max_position_value_cny)
        self.lot_size = int(lot_size)
        self.state = RiskState()

    def trades_left(self) -> int:
        self.state.reset_if_new_day()
        return max(0, self.max_trades_per_day - self.state.trades_today)

    def can_trade(
        self,
        signal: TradeSignal,
        *,
        account: AccountState,
        snapshot: MarketSnapshot,
        suggested_shares: int,
        lot_size: int,
    ) -> tuple[bool, str]:
        """
        Shares-based risk checks (CN-A lot rules).
        """
        self.state.reset_if_new_day()

        if self.state.trades_today >= self.max_trades_per_day:
            return False, "trade_limit_reached"

        if float(signal.confidence) < self.min_confidence:
            return False, "confidence_too_low"

        if self.state.last_trade_ts is not None:
            dt = (datetime.now(CN_TZ) - self.state.last_trade_ts).total_seconds()
            if dt < self.cooldown_seconds:
                return False, "cooldown_active"

        if signal.action == "HOLD":
            return False, "hold_signal"

        last = float(snapshot.last_price or 0.0)
        if last <= 0:
            return False, "invalid_last_price"

        cash = float(account.cash_cny)
        pos = int(account.position_shares)

        lot_size = int(lot_size) if lot_size and int(lot_size) > 0 else int(self.lot_size)

        # must be positive and aligned to lot size
        shares = int(suggested_shares)
        if shares <= 0:
            return False, "shares_zero"
        if shares % lot_size != 0:
            return False, "shares_not_multiple_of_lot"

        # exposure checks
        position_value_cny = pos * last
        if position_value_cny > self.max_position_value_cny * 1.001:
            return False, "max_position_value_exceeded"

        order_notional = shares * last

        if signal.action == "BUY":
            if cash < order_notional:
                return False, "insufficient_cash"
            if (position_value_cny + order_notional) > self.max_position_value_cny:
                return False, "would_exceed_max_position_value"

        if signal.action == "SELL":
            if pos <= 0:
                return False, "no_position_to_sell"
            if shares > pos:
                return False, "sell_shares_exceed_position"

        return True, "ok"

    def record_trade(self):
        self.state.reset_if_new_day()
        self.state.trades_today += 1
        self.state.last_trade_ts = datetime.now(CN_TZ)
