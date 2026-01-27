# scripts/migrate_intraday_trades_add_user_account.py
from __future__ import annotations

import sys
from pathlib import Path
import asyncio

from sqlalchemy import text

# Ensure project root is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.config import settings
from app.storage.db import make_engine, init_sqlite_pragmas

TABLE = "intraday_trades"

# SQLite-safe: add columns with explicit defaults so old rows are valid
COLUMNS = [
    ("user_id", "INTEGER NOT NULL DEFAULT 0"),
    ("username", "TEXT NOT NULL DEFAULT ''"),
    ("account_id", "TEXT NOT NULL DEFAULT 'default'"),
]

# Optional indexes that match the new query patterns
INDEXES = [
    (
        "ix_intraday_trades_user_account_symbol_day_ts",
        "(user_id, account_id, symbol, trading_day_sh, decision_ts)",
    ),
    ("ix_intraday_trades_user_id", "(user_id)"),
    ("ix_intraday_trades_account_id", "(account_id)"),
]


async def _existing_columns(conn) -> set[str]:
    res = await conn.execute(text(f"PRAGMA table_info({TABLE})"))
    return {row[1] for row in res.fetchall()}  # row[1] is column name


async def _existing_indexes(conn) -> set[str]:
    res = await conn.execute(text(f"PRAGMA index_list({TABLE})"))
    return {row[1] for row in res.fetchall()}  # row[1] is index name


async def main() -> None:
    engine = make_engine(settings.database_url)
    try:
        # Make sure pragmas are applied (WAL/FK)
        await init_sqlite_pragmas(engine)

        async with engine.begin() as conn:
            existing_cols = await _existing_columns(conn)

            for name, ddl in COLUMNS:
                if name in existing_cols:
                    print(f"Skip column (exists): {name}")
                    continue
                await conn.execute(text(f"ALTER TABLE {TABLE} ADD COLUMN {name} {ddl}"))
                print(f"Added column: {name} {ddl}")

            existing_idx = await _existing_indexes(conn)
            for idx_name, idx_cols in INDEXES:
                if idx_name in existing_idx:
                    print(f"Skip index (exists): {idx_name}")
                    continue
                await conn.execute(text(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {TABLE} {idx_cols}"))
                print(f"Ensured index: {idx_name} ON {idx_cols}")

        print("Migration done.")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
