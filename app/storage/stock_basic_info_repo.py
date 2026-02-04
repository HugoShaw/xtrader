# app/storage/stock_basic_info_repo.py
from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.sqlite import insert

from app.storage.orm_models import StockBasicInfoORM, StockBasicInfoFetchORM


class StockBasicInfoRepo:
    def __init__(self, session: AsyncSession):
        self.s = session

    async def get_info(self, *, symbol: str) -> Optional[StockBasicInfoORM]:
        stmt = select(StockBasicInfoORM).where(StockBasicInfoORM.symbol == symbol)
        return (await self.s.execute(stmt)).scalars().first()

    async def upsert_info(self, *, symbol: str, source: str, data_json: str) -> None:
        stmt = insert(StockBasicInfoORM).values(
            {
                "symbol": symbol,
                "source": source,
                "data_json": data_json,
                "updated_at": datetime.utcnow(),
            }
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"],
            set_={
                "source": stmt.excluded.source,
                "data_json": stmt.excluded.data_json,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        await self.s.execute(stmt)

    async def get_fetch_log(
        self,
        *,
        user_id: int,
        symbol: str,
        fetch_date: date,
    ) -> Optional[StockBasicInfoFetchORM]:
        stmt = (
            select(StockBasicInfoFetchORM)
            .where(StockBasicInfoFetchORM.user_id == int(user_id))
            .where(StockBasicInfoFetchORM.symbol == symbol)
            .where(StockBasicInfoFetchORM.fetch_date == fetch_date)
        )
        return (await self.s.execute(stmt)).scalars().first()

    async def add_fetch_log(
        self,
        *,
        user_id: int,
        symbol: str,
        fetch_date: date,
    ) -> None:
        stmt = insert(StockBasicInfoFetchORM).values(
            {
                "user_id": int(user_id),
                "symbol": symbol,
                "fetch_date": fetch_date,
                "created_at": datetime.utcnow(),
            }
        )
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["user_id", "symbol", "fetch_date"],
        )
        await self.s.execute(stmt)