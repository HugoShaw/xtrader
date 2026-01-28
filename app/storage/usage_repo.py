from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage.orm_models import ApiUsageORM


class ApiUsageRepo:
    def __init__(self, session: AsyncSession):
        self.s = session

    async def add(self, *, path: str, method: str, status_code: int, username: Optional[str], duration_ms: Optional[float]) -> None:
        row = ApiUsageORM(
            path=str(path),
            method=str(method).upper(),
            status_code=int(status_code),
            username=str(username) if username else None,
            duration_ms=float(duration_ms) if duration_ms is not None else None,
        )
        self.s.add(row)

    async def list_daily_counts(self, *, days: int = 7) -> list[dict]:
        days = max(1, int(days))
        start_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)
        stmt = (
            select(func.date(ApiUsageORM.created_at).label("day"), func.count(ApiUsageORM.id).label("count"))
            .where(ApiUsageORM.created_at >= start_dt)
            .group_by("day")
            .order_by("day")
        )
        rows = (await self.s.execute(stmt)).all()
        return [{"date": str(day), "count": int(count or 0)} for day, count in rows]

    async def top_paths(self, *, days: int = 7, limit: int = 8) -> list[dict]:
        days = max(1, int(days))
        start_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)
        stmt = (
            select(ApiUsageORM.path, func.count(ApiUsageORM.id).label("count"))
            .where(ApiUsageORM.created_at >= start_dt)
            .group_by(ApiUsageORM.path)
            .order_by(desc(func.count(ApiUsageORM.id)))
            .limit(int(limit))
        )
        rows = (await self.s.execute(stmt)).all()
        return [{"path": str(path), "count": int(count or 0)} for path, count in rows]

    async def top_users(self, *, days: int = 7, limit: int = 8) -> list[dict]:
        days = max(1, int(days))
        start_dt = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)
        stmt = (
            select(ApiUsageORM.username, func.count(ApiUsageORM.id).label("count"))
            .where(ApiUsageORM.created_at >= start_dt)
            .where(ApiUsageORM.username.is_not(None))
            .group_by(ApiUsageORM.username)
            .order_by(desc(func.count(ApiUsageORM.id)))
            .limit(int(limit))
        )
        rows = (await self.s.execute(stmt)).all()
        return [{"username": str(username), "count": int(count or 0)} for username, count in rows]
