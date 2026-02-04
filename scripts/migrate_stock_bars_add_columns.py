# scripts/migrate_stock_bars_add_columns.py
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

TABLE = "stock_bars"

# SQLite-safe: add columns with NULL default to preserve existing rows
COLUMNS = [
    ("open", "REAL"),
    ("high", "REAL"),
    ("low", "REAL"),
    ("close", "REAL"),
    ("volume", "REAL"),
    ("amount", "REAL"),
    ("vwap", "REAL"),
    ("amplitude", "REAL"),
    ("pct_change", "REAL"),
    ("change", "REAL"),
    ("turnover", "REAL"),
]


async def _existing_columns(conn) -> set[str]:
    res = await conn.execute(text(f"PRAGMA table_info({TABLE})"))
    return {row[1] for row in res.fetchall()}  # row[1] is column name


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

        print("Migration done.")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
