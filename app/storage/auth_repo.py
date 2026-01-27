from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage.orm_models import UserORM


class UserRepo:
    def __init__(self, session: AsyncSession):
        self.s = session

    async def get_by_username(self, username: str) -> Optional[UserORM]:
        stmt = select(UserORM).where(UserORM.username == username)
        return (await self.s.execute(stmt)).scalars().first()

    async def create_user(self, *, username: str, password_hash: str, salt: str) -> UserORM:
        user = UserORM(username=username, password_hash=password_hash, salt=salt)
        self.s.add(user)
        await self.s.flush()
        return user

    async def update_last_login(self, user: UserORM) -> None:
        user.last_login_at = datetime.utcnow()
        self.s.add(user)

