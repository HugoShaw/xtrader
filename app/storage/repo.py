# app/storage/repo.py
from __future__ import annotations

from datetime import datetime, date
from typing import Any, Dict, List, Optional

from sqlalchemy import select, delete, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage.orm_models import IntradayTradeORM


class IntradayTradeRepo:
    def __init__(self, session: AsyncSession):
        self.s = session

    async def add(self, row: Dict[str, Any]) -> None:
        self.s.add(IntradayTradeORM(**row))

    async def list_by_day(
        self,
        symbol: str,
        trading_day_sh: date,
        *,
        limit: int = 50,
        asc: bool = True,
    ) -> List[IntradayTradeORM]:
        order = IntradayTradeORM.decision_ts.asc() if asc else IntradayTradeORM.decision_ts.desc()
        stmt = (
            select(IntradayTradeORM)
            .where(IntradayTradeORM.symbol == symbol)
            .where(IntradayTradeORM.trading_day_sh == trading_day_sh)
            .order_by(order)
            .limit(int(limit))
        )
        return (await self.s.execute(stmt)).scalars().all()

    async def list_recent(
        self,
        symbol: str,
        *,
        limit: int = 50,
    ) -> List[IntradayTradeORM]:
        stmt = (
            select(IntradayTradeORM)
            .where(IntradayTradeORM.symbol == symbol)
            .order_by(desc(IntradayTradeORM.decision_ts))
            .limit(int(limit))
        )
        return (await self.s.execute(stmt)).scalars().all()

    async def clear_symbol(self, symbol: str) -> int:
        stmt = delete(IntradayTradeORM).where(IntradayTradeORM.symbol == symbol)
        res = await self.s.execute(stmt)
        return int(res.rowcount or 0)
