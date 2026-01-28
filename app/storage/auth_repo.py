from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage.orm_models import UserORM


class UserRepo:
    def __init__(self, session: AsyncSession):
        self.s = session

    async def get_by_username(self, username: str) -> Optional[UserORM]:
        stmt = select(UserORM).where(UserORM.username == username)
        return (await self.s.execute(stmt)).scalars().first()

    async def get_by_id(self, user_id: int) -> Optional[UserORM]:
        stmt = select(UserORM).where(UserORM.id == int(user_id))
        return (await self.s.execute(stmt)).scalars().first()

    async def create_user(self, *, username: str, password_hash: str, salt: str) -> UserORM:
        user = UserORM(username=username, password_hash=password_hash, salt=salt)
        self.s.add(user)
        await self.s.flush()
        return user

    async def update_last_login(self, user: UserORM) -> None:
        user.last_login_at = datetime.utcnow()
        self.s.add(user)

    async def list_users(self, *, limit: int = 50, offset: int = 0, search: Optional[str] = None) -> list[UserORM]:
        stmt = select(UserORM)
        if search:
            stmt = stmt.where(UserORM.username.like(f"%{search}%"))
        stmt = stmt.order_by(UserORM.id.asc()).limit(int(limit)).offset(int(offset))
        return (await self.s.execute(stmt)).scalars().all()

    async def count_users(self, *, search: Optional[str] = None) -> int:
        stmt = select(func.count(UserORM.id))
        if search:
            stmt = stmt.where(UserORM.username.like(f"%{search}%"))
        res = await self.s.execute(stmt)
        return int(res.scalar() or 0)

    async def delete_user(self, user_id: int) -> int:
        stmt = delete(UserORM).where(UserORM.id == int(user_id))
        res = await self.s.execute(stmt)
        return int(res.rowcount or 0)
