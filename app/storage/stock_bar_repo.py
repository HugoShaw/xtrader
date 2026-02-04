# app/storage/stock_bar_repo.py
from __future__ import annotations

from datetime import datetime, date
from typing import List, Optional, Sequence

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.sqlite import insert

from app.storage.orm_models import StockBarORM, StockBarFetchORM


class StockBarRepo:
    def __init__(self, session: AsyncSession):
        self.s = session

    async def get_range(
        self,
        *,
        symbol: str,
        period: str,
        adjust: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> List[StockBarORM]:
        stmt = (
            select(StockBarORM)
            .where(StockBarORM.symbol == symbol)
            .where(StockBarORM.period == period)
            .where(StockBarORM.adjust == adjust)
            .where(StockBarORM.ts >= start_dt)
            .where(StockBarORM.ts <= end_dt)
            .order_by(StockBarORM.ts.asc())
        )
        return (await self.s.execute(stmt)).scalars().all()

    async def get_min_max_ts(
        self,
        *,
        symbol: str,
        period: str,
        adjust: str,
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        stmt = (
            select(func.min(StockBarORM.ts), func.max(StockBarORM.ts))
            .where(StockBarORM.symbol == symbol)
            .where(StockBarORM.period == period)
            .where(StockBarORM.adjust == adjust)
        )
        row = (await self.s.execute(stmt)).one()
        return row[0], row[1]

    async def get_max_ts(
        self,
        *,
        symbol: str,
        period: str,
        adjust: str,
    ) -> Optional[datetime]:
        stmt = (
            select(func.max(StockBarORM.ts))
            .where(StockBarORM.symbol == symbol)
            .where(StockBarORM.period == period)
            .where(StockBarORM.adjust == adjust)
        )
        return (await self.s.execute(stmt)).scalar_one_or_none()

    async def upsert_bars(self, rows: Sequence[dict]) -> None:
        if not rows:
            return
        stmt = insert(StockBarORM).values(list(rows))
        update_cols = {
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
            "amount": stmt.excluded.amount,
            "vwap": stmt.excluded.vwap,
            "amplitude": stmt.excluded.amplitude,
            "pct_change": stmt.excluded.pct_change,
            "change": stmt.excluded.change,
            "turnover": stmt.excluded.turnover,
            "updated_at": stmt.excluded.updated_at,
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "period", "adjust", "ts"],
            set_=update_cols,
        )
        await self.s.execute(stmt)

    async def get_fetch_log(
        self,
        *,
        symbol: str,
        period: str,
        adjust: str,
    ) -> Optional[StockBarFetchORM]:
        stmt = (
            select(StockBarFetchORM)
            .where(StockBarFetchORM.symbol == symbol)
            .where(StockBarFetchORM.period == period)
            .where(StockBarFetchORM.adjust == adjust)
        )
        return (await self.s.execute(stmt)).scalars().first()

    async def upsert_fetch_log(
        self,
        *,
        symbol: str,
        period: str,
        adjust: str,
        fetch_date: date,
        fetch_start: Optional[str],
        fetch_end: Optional[str],
    ) -> None:
        stmt = insert(StockBarFetchORM).values(
            {
                "symbol": symbol,
                "period": period,
                "adjust": adjust,
                "last_fetch_date": fetch_date,
                "last_fetch_start": fetch_start,
                "last_fetch_end": fetch_end,
                "updated_at": datetime.utcnow(),
            }
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "period", "adjust"],
            set_={
                "last_fetch_date": stmt.excluded.last_fetch_date,
                "last_fetch_start": stmt.excluded.last_fetch_start,
                "last_fetch_end": stmt.excluded.last_fetch_end,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        await self.s.execute(stmt)
