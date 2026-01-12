# app/services/market_data_dummy.py
from __future__ import annotations

from typing import Any 

from datetime import datetime, timezone

from app.models import MarketSnapshot, Bar 

class DummyMarketDataProvider:
    async def get_snapshot(self, symbol: str, **kwargs: Any) -> MarketSnapshot:
        now = datetime.now(timezone.utc)
        return MarketSnapshot(
            symbol=symbol,
            ts=now,
            last_price=100.0,
            day_open=99.0,
            day_high=101.0,
            day_low=98.5,
            day_volume=1_000_000,
            vwap=99.8,
            prev_close=99.5,
            recent_bars=[Bar(ts=now, open=99.9, high=100.2, low=99.8, close=100.0, volume=12000)],
            extra={"note": "dummy provider"},
        )