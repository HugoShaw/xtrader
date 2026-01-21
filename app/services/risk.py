# app/services/risk.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Tuple

from app.models import TradeSignal, AccountState, MarketSnapshot, ExecutionConstraints
from app.utils.timeutils import now_shanghai, ensure_shanghai


@dataclass
class RiskState:
    trades_today: int = 0
    last_trade_ts: Optional[datetime] = None  # tz-aware Shanghai
    day: date = now_shanghai().date()

    def reset_if_new_day(self, *, ref_dt: Optional[datetime] = None) -> None:
        """
        Reset counters when entering a new Shanghai trading day.
        Prefer using the decision timestamp (snapshot.ts) if supplied.
        """
        dt = ensure_shanghai(ref_dt) if ref_dt is not None else ensure_shanghai(now_shanghai())
        today = dt.date()
        if today != self.day:
            self.day = today
            self.trades_today = 0
            self.last_trade_ts = None


def _safe_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if (v == v) else None  # NaN guard
    except Exception:
        return None


def _all_in_cost_rate_from_constraints(constraints: ExecutionConstraints) -> float:
    """
    Option A: treat total friction as fee + slippage, both in bps (1e-4).
    """
    fee_bps = int(getattr(constraints, "fees_bps_est", 0) or 0)
    slip_bps = int(getattr(constraints, "slippage_bps_est", 0) or 0)
    # clamp to sane bounds to avoid user mistakes
    fee_bps = max(0, min(fee_bps, 50_000))
    slip_bps = max(0, min(slip_bps, 50_000))
    return (fee_bps + slip_bps) / 1e4


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
        self.max_trades_per_day = max(0, int(max_trades_per_day))
        self.cooldown_seconds = max(0, int(cooldown_seconds))
        self.min_confidence = float(min_confidence)
        self.max_position_value_cny = max(0.0, float(max_position_value_cny))
        self.lot_size = max(1, int(lot_size))
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
        constraints: Optional[ExecutionConstraints] = None,
        suggested_shares: int,
        lot_size: int,
        decision_dt: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """
        Shares-based risk checks (CN-A lot rules).
        Uses Shanghai time; cooldown anchored to decision_dt (snapshot.ts) if possible.

        If constraints is supplied:
          - BUY cash check includes all-in friction (fee+slippage) using bps in constraints.
        """
        # anchor day/cooldown to decision time when provided
        if decision_dt is None:
            decision_dt = getattr(snapshot, "ts", None)
        decision_dt = ensure_shanghai(decision_dt) if decision_dt is not None else ensure_shanghai(now_shanghai())

        self.state.reset_if_new_day(ref_dt=decision_dt)

        if signal.action == "HOLD":
            return False, "hold_signal"

        if self.state.trades_today >= self.max_trades_per_day:
            return False, "trade_limit_reached"

        conf = _safe_float(getattr(signal, "confidence", None)) or 0.0
        if conf < float(self.min_confidence):
            return False, "confidence_too_low"

        # cooldown
        if self.state.last_trade_ts is not None and self.cooldown_seconds > 0:
            last_dt = ensure_shanghai(self.state.last_trade_ts)
            dt_sec = (decision_dt - last_dt).total_seconds()
            if dt_sec < float(self.cooldown_seconds):
                return False, "cooldown_active"

        last = _safe_float(getattr(snapshot, "last_price", None)) or 0.0
        if last <= 0:
            return False, "invalid_last_price"

        cash = float(getattr(account, "cash_cny", 0.0) or 0.0)
        pos = int(getattr(account, "position_shares", 0) or 0)

        lot_size_eff = int(lot_size) if lot_size and int(lot_size) > 0 else int(self.lot_size)

        shares = int(suggested_shares or 0)
        if shares <= 0:
            return False, "shares_zero"
        if shares % lot_size_eff != 0:
            return False, "shares_not_multiple_of_lot"

        position_value_cny = float(pos) * float(last)

        # small tolerance to avoid float edge rejects
        if position_value_cny > self.max_position_value_cny * 1.001:
            return False, "max_position_value_exceeded"

        order_notional = float(shares) * float(last)

        # all-in friction for cash impact (Option A)
        cost_rate = 0.0
        if constraints is not None:
            cost_rate = _all_in_cost_rate_from_constraints(constraints)

        if signal.action == "BUY":
            # cash impact includes costs (fees+slippage)
            cash_required = order_notional * (1.0 + cost_rate)
            if cash < cash_required - 1e-6:
                return False, "insufficient_cash"
            if (position_value_cny + order_notional) > self.max_position_value_cny + 1e-6:
                return False, "would_exceed_max_position_value"

        if signal.action == "SELL":
            if pos <= 0:
                return False, "no_position_to_sell"
            if shares > pos:
                return False, "sell_shares_exceed_position"

        return True, "ok"

    def record_trade(self, *, decision_dt: Optional[datetime] = None) -> None:
        """
        Record a successful execution for trade limit + cooldown.
        Prefer using the same decision_dt you used in can_trade().
        """
        if decision_dt is None:
            decision_dt = now_shanghai()
        decision_dt = ensure_shanghai(decision_dt)

        self.state.reset_if_new_day(ref_dt=decision_dt)
        self.state.trades_today += 1
        self.state.last_trade_ts = decision_dt
