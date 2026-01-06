# app/storage/db.py
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

def make_engine(database_url: str):
    # SQLite production notes:
    # - WAL improves concurrency
    # - foreign_keys=ON
    engine = create_async_engine(
        database_url,
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},
    )
    return engine

def make_session_factory(engine):
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        class_=AsyncSession,
        autoflush=False,
        autocommit=False,
    )

async def init_sqlite_pragmas(engine) -> None:
    # WAL + FK enforcement
    async with engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
        await conn.exec_driver_sql("PRAGMA foreign_keys=ON;")
        await conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")

async def init_db(engine) -> None:
    # Import models so metadata is populated
    from app.storage import orm_models  # noqa: F401

    await init_sqlite_pragmas(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@asynccontextmanager
async def session_scope(session_factory) -> AsyncIterator[AsyncSession]:
    session: AsyncSession = session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
