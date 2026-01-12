# app/services/broker.py
from __future__ import annotations

from typing import Protocol
from datetime import datetime, timezone

from app.models import TradeRequest, TradeResult


class Broker(Protocol):
    async def place_order(self, req: TradeRequest) -> TradeResult: ...


class PaperBroker:
    async def place_order(self, req: TradeRequest) -> TradeResult:
        lots = int(req.shares // req.lot_size) if req.lot_size else 0
        return TradeResult(
            ok=True,
            message="paper order accepted",
            ts=datetime.now(timezone.utc),
            symbol=req.symbol,
            action=req.action,
            shares=int(req.shares),
            lots=lots,
            paper=True,
            details={"fill": "simulated"},
        )
