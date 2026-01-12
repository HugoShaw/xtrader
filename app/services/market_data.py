# app/services/market_data.py
from __future__ import annotations

from typing import Protocol, Any

from app.models import MarketSnapshot
from app.services.cache import AsyncCache
from app.services.market_data_akshare import AkShareAStockProvider


class MarketDataProvider(Protocol):
    async def get_snapshot(self, symbol: str, **kwargs: Any) -> MarketSnapshot: ...


def build_market_provider(
    name: str,
    *,
    cache: AsyncCache,
    # spot_ttl_sec: float = 3.0,
    orderbook_ttl_sec: float = 1.0,
    bars_ttl_sec: float = 2.0,
) -> MarketDataProvider:
    n = (name or "").strip().lower()
    if n in {"akshare", "eastmoney", "china_a", "china-a", "a_share"}:
        return AkShareAStockProvider(
            cache=cache,
            # spot_ttl_sec=spot_ttl_sec,
            orderbook_ttl_sec=orderbook_ttl_sec,
            bars_ttl_sec=bars_ttl_sec,
        )

    # fallback: keep dummy provider if you want it
    from app.services.market_data_dummy import DummyMarketDataProvider  # optional
    return DummyMarketDataProvider()
