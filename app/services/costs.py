# app/services/costs.py
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Dict

from app.config import settings

# High precision for money math
getcontext().prec = 28


def _d(x) -> Decimal:
    """Safe Decimal conversion"""
    return Decimal(str(x))


@dataclass(frozen=True)
class CostModel:
    """
    Canonical trading cost model.
    SINGLE source of truth used by:
      - strategy
      - risk
      - broker
      - backtest
    """

    buy_fee_rate: Decimal
    sell_fee_rate: Decimal
    slippage_bps: int = 0   # optional, can move to config later

    # -----------------------
    # factory
    # -----------------------
    @classmethod
    def from_settings(cls) -> "CostModel":
        return cls(
            buy_fee_rate=_d(settings.buy_fee_rate),
            sell_fee_rate=_d(settings.sell_fee_rate),
            slippage_bps=0,
        )

    # -----------------------
    # helpers
    # -----------------------
    def fee_rate(self, side: str) -> Decimal:
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"Invalid side: {side}")
        return self.buy_fee_rate if side == "BUY" else self.sell_fee_rate

    def round_trip_cost_bps(self) -> int:
        """
        Approx round-trip cost in bps:
        BUY fee + SELL fee + 2 * slippage
        """
        fee_bps = (self.buy_fee_rate + self.sell_fee_rate) * Decimal(10_000)
        return int(fee_bps) + 2 * self.slippage_bps

    # -----------------------
    # main API
    # -----------------------
    def estimate(
        self,
        *,
        side: str,
        price: float | Decimal,
        shares: int,
    ) -> Dict[str, Decimal]:
        """
        Returns:
          notional
          fee
          slippage
          total_cost
          effective_price
          cash_delta
        """

        price = _d(price)
        shares_d = Decimal(shares)

        notional = price * shares_d

        fee = notional * self.fee_rate(side)
        slippage = (
            notional * Decimal(self.slippage_bps) / Decimal(10_000)
            if self.slippage_bps > 0
            else Decimal("0")
        )

        total_cost = fee + slippage

        if side == "BUY":
            effective_price = (notional + total_cost) / shares_d
            cash_delta = -(notional + total_cost)
        else:  # SELL
            effective_price = (notional - total_cost) / shares_d
            cash_delta = +(notional - total_cost)

        return {
            "notional": notional,
            "fee": fee,
            "slippage": slippage,
            "total_cost": total_cost,
            "effective_price": effective_price,
            "cash_delta": cash_delta,
        }
