from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage.orm_models import TradeAccountORM, TradePositionORM


class TradeAccountRepo:
    def __init__(self, session: AsyncSession):
        self.s = session

    async def list_accounts(self, *, user_id: int) -> List[TradeAccountORM]:
        stmt = (
            select(TradeAccountORM)
            .where(TradeAccountORM.user_id == int(user_id))
            .order_by(TradeAccountORM.created_at.desc())
        )
        return (await self.s.execute(stmt)).scalars().all()

    async def get_account(self, *, user_id: int, account_id: str) -> Optional[TradeAccountORM]:
        stmt = (
            select(TradeAccountORM)
            .where(TradeAccountORM.user_id == int(user_id))
            .where(TradeAccountORM.account_id == str(account_id))
        )
        return (await self.s.execute(stmt)).scalars().first()

    async def create_account(
        self,
        *,
        user_id: int,
        account_id: str,
        name: Optional[str],
        cash_cny: float,
        base_currency: str,
    ) -> TradeAccountORM:
        now = datetime.utcnow()
        acc = TradeAccountORM(
            user_id=int(user_id),
            account_id=str(account_id),
            name=name,
            cash_cny=float(cash_cny),
            base_currency=str(base_currency),
            created_at=now,
            updated_at=now,
        )
        self.s.add(acc)
        await self.s.flush()
        return acc

    async def update_account(
        self,
        acc: TradeAccountORM,
        *,
        name: Optional[str],
        cash_cny: Optional[float],
        base_currency: Optional[str],
    ) -> TradeAccountORM:
        if name is not None:
            acc.name = name
        if cash_cny is not None:
            acc.cash_cny = float(cash_cny)
        if base_currency is not None:
            acc.base_currency = str(base_currency)
        acc.updated_at = datetime.utcnow()
        self.s.add(acc)
        return acc

    async def delete_account(self, *, account_pk: int, user_id: int) -> int:
        await self.s.execute(
            delete(TradePositionORM)
            .where(TradePositionORM.user_id == int(user_id))
            .where(TradePositionORM.account_pk == int(account_pk))
        )
        res = await self.s.execute(
            delete(TradeAccountORM)
            .where(TradeAccountORM.user_id == int(user_id))
            .where(TradeAccountORM.id == int(account_pk))
        )
        return int(res.rowcount or 0)

    async def list_positions(self, *, account_pk: int, user_id: int) -> List[TradePositionORM]:
        stmt = (
            select(TradePositionORM)
            .where(TradePositionORM.user_id == int(user_id))
            .where(TradePositionORM.account_pk == int(account_pk))
            .order_by(TradePositionORM.symbol.asc())
        )
        return (await self.s.execute(stmt)).scalars().all()

    async def get_position(
        self,
        *,
        account_pk: int,
        user_id: int,
        symbol: str,
    ) -> Optional[TradePositionORM]:
        stmt = (
            select(TradePositionORM)
            .where(TradePositionORM.user_id == int(user_id))
            .where(TradePositionORM.account_pk == int(account_pk))
            .where(TradePositionORM.symbol == str(symbol))
        )
        return (await self.s.execute(stmt)).scalars().first()

    async def create_position(
        self,
        *,
        account_pk: int,
        user_id: int,
        symbol: str,
        shares: int,
        avg_cost_cny: Optional[float],
        unrealized_pnl_cny: Optional[float],
    ) -> TradePositionORM:
        now = datetime.utcnow()
        pos = TradePositionORM(
            account_pk=int(account_pk),
            user_id=int(user_id),
            symbol=str(symbol),
            shares=int(shares),
            avg_cost_cny=avg_cost_cny,
            unrealized_pnl_cny=unrealized_pnl_cny,
            created_at=now,
            updated_at=now,
        )
        self.s.add(pos)
        await self.s.flush()
        return pos

    async def update_position(
        self,
        pos: TradePositionORM,
        *,
        shares: Optional[int],
        avg_cost_cny: Optional[float],
        unrealized_pnl_cny: Optional[float],
    ) -> TradePositionORM:
        if shares is not None:
            pos.shares = int(shares)
        if avg_cost_cny is not None:
            pos.avg_cost_cny = avg_cost_cny
        if unrealized_pnl_cny is not None:
            pos.unrealized_pnl_cny = unrealized_pnl_cny
        pos.updated_at = datetime.utcnow()
        self.s.add(pos)
        return pos

    async def delete_position(self, *, account_pk: int, user_id: int, symbol: str) -> int:
        res = await self.s.execute(
            delete(TradePositionORM)
            .where(TradePositionORM.user_id == int(user_id))
            .where(TradePositionORM.account_pk == int(account_pk))
            .where(TradePositionORM.symbol == str(symbol))
        )
        return int(res.rowcount or 0)
