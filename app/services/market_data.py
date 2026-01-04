# app/services/market_data.py
from __future__ import annotations
from typing import Protocol
from datetime import datetime, timezone
from app.models import MarketSnapshot, Bar
from app.services.market_data_akshare import AkShareAStockProvider

class MarketDataProvider(Protocol):
    async def get_snapshot(self, symbol: str) -> MarketSnapshot: ...

class DummyMarketDataProvider:
    """Replace with Polygon/AlphaVantage/Broker implementation."""
    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        now = datetime.now(timezone.utc)
        # TODO: fetch real snapshot + recent bars
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
            recent_bars=[
                Bar(ts=now, open=99.9, high=100.2, low=99.8, close=100.0, volume=12000)
            ],
            extra={"note": "dummy provider"}
        )

def build_market_provider(name: str):
    if name.lower() in {"akshare", "eastmoney", "china_a"}:
        return AkShareAStockProvider()
    # fallback
    return DummyMarketDataProvider()