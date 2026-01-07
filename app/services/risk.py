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
        suggested_notional_cny: float,
    ) -> tuple[bool, str]:
        """
        NOTE: this checks only risk/eligibility.
        Actual share sizing can still happen in Strategy/Broker, but this blocks obvious violations.
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

        # Current exposure
        position_value_cny = pos * last
        if position_value_cny > self.max_position_value_cny * 1.001:
            return False, "max_position_value_exceeded"

        notional = max(0.0, float(suggested_notional_cny))

        # Optional: block “too small to execute 1 lot” (helps reduce nonsense trades)
        min_lot_value = self.lot_size * last
        if notional > 0 and notional < min_lot_value * 0.98:
            return False, "notional_too_small_for_one_lot"

        if signal.action == "BUY":
            # basic cash check
            if notional <= 0:
                return False, "buy_notional_zero"
            if cash < notional:
                return False, "insufficient_cash"

            # exposure after buy (approx)
            if (position_value_cny + notional) > self.max_position_value_cny:
                return False, "would_exceed_max_position_value"

        if signal.action == "SELL":
            if pos <= 0:
                return False, "no_position_to_sell"
            # if you later compute shares precisely, also ensure shares_to_sell <= pos

        return True, "ok"

    def record_trade(self):
        self.state.reset_if_new_day()
        self.state.trades_today += 1
        self.state.last_trade_ts = datetime.now(CN_TZ)
