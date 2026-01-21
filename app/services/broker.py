# app/services/broker.py
from __future__ import annotations

from typing import Protocol

from app.models import TradeRequest, TradeResult
from app.services.costs import CostModel
from app.utils.timeutils import now_shanghai, ensure_shanghai


class Broker(Protocol):
    async def place_order(self, req: TradeRequest) -> TradeResult: ...


class PaperBroker:
    def __init__(self) -> None:
        self.costs = CostModel.from_settings()

    async def place_order(self, req: TradeRequest) -> TradeResult:
        ts = ensure_shanghai(now_shanghai())
        shares = int(req.shares)
        lot_size = int(req.lot_size) if req.lot_size else 0
        lots = shares // lot_size if lot_size else 0

        details = {"fill": "simulated"}

        # No price -> accept but no cost computed
        if req.price_cny is None:
            details["warning_price"] = "price_cny missing; cost fields not computed"
            return TradeResult(
                ok=True,
                message="paper order accepted (no price; no cost computed)",
                ts=ts,
                symbol=req.symbol,
                action=req.action,
                shares=shares,
                lots=lots,
                paper=True,
                details=details,
            )

        executed_price = float(req.price_cny)

        est = self.costs.estimate(
            side=req.action,
            price=executed_price,
            shares=shares,
        )

        notional = float(est["notional"])
        fee = float(est["fee"])
        slippage = float(est["slippage"])
        total_cost = float(est["total_cost"])
        effective_price = float(est["effective_price"])
        cash_delta = float(est["cash_delta"])

        # legacy fees_cny = total_cost_cny (all-in friction)
        fees_legacy = total_cost

        details["cost_model"] = {
            "buy_fee_rate": float(self.costs.buy_fee_rate),
            "sell_fee_rate": float(self.costs.sell_fee_rate),
            "slippage_bps": int(self.costs.slippage_bps),
            "fees_cny_definition": "fees_cny == total_cost_cny (fee + slippage)",
        }

        return TradeResult(
            ok=True,
            message="paper order accepted",
            ts=ts,
            symbol=req.symbol,
            action=req.action,
            shares=shares,
            lots=lots,
            paper=True,

            executed_price_cny=executed_price,
            executed_price_net_cny=effective_price,
            notional_cny=notional,

            # breakdown
            fee_cny=fee,
            slippage_cny=slippage,
            total_cost_cny=total_cost,
            cash_delta_cny=cash_delta,

            # ✅ backward compatible
            fees_cny=fees_legacy,

            details=details,
        )