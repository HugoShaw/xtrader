# scripts/migrate_intraday_trades_add_costs.py
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

COLUMNS = [
    ("executed_price_net_cny", "FLOAT"),
    ("notional_cny", "FLOAT"),
    ("fee_cny", "FLOAT"),
    ("slippage_cny", "FLOAT"),
    ("total_cost_cny", "FLOAT"),
    ("cash_delta_cny", "FLOAT"),
]

TABLE = "intraday_trades"


async def main() -> None:
    engine = make_engine(settings.database_url)
    try:
        # Make sure pragmas are applied (WAL/FK)
        await init_sqlite_pragmas(engine)

        async with engine.begin() as conn:
            res = await conn.execute(text(f"PRAGMA table_info({TABLE})"))
            existing = {row[1] for row in res.fetchall()}  # row[1] is column name

            for name, typ in COLUMNS:
                if name in existing:
                    print(f"Skip (exists): {name}")
                    continue
                await conn.execute(text(f"ALTER TABLE {TABLE} ADD COLUMN {name} {typ}"))
                print(f"Added column: {name}")

        print("Migration done.")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
