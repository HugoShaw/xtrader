# app/services/broker.py
from __future__ import annotations
from typing import Protocol
from datetime import datetime, timezone
from app.models import TradeRequest, TradeResult

class Broker(Protocol):
    async def place_order(self, req: TradeRequest) -> TradeResult: ...

class PaperBroker:
    async def place_order(self, req: TradeRequest) -> TradeResult:
        # TODO: log to DB, simulate fills, slippage, fees, etc.
        return TradeResult(
            ok=True,
            message="paper order accepted",
            ts=datetime.now(timezone.utc),
            symbol=req.symbol,
            action=req.action,
            notional_cny=req.notional_cny,
            paper=True,
            details={"fill": "simulated"}
        )
