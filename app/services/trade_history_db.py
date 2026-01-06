# app/services/trade_history_db.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.storage.db import session_scope
from app.storage.repo import TradeFeedbackRepo


class TradeHistoryDB:
    """
    DB-backed trade history used by prompt_builder.
    """
    def __init__(self, session_factory, *, max_records: int = 50):
        self.session_factory = session_factory
        self.max_records = max_records

    async def list_records(self, symbol: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        limit = min(int(limit or self.max_records), self.max_records)
        async with session_scope(self.session_factory) as s:
            repo = TradeFeedbackRepo(s)
            return await repo.list_recent(symbol, limit=limit)

    async def append_record(self, symbol: str, record: Dict[str, Any]) -> None:
        """
        record format should match your prompt 'trade_history.records' element.
        """
        # parse decision_ts
        decision_ts_str = record.get("decision_ts")
        if not decision_ts_str:
            return
        decision_ts = datetime.strptime(decision_ts_str, "%Y-%m-%d %H:%M:%S")

        async with session_scope(self.session_factory) as s:
            repo = TradeFeedbackRepo(s)
            await repo.add_feedback(
                symbol=symbol,
                decision_ts=decision_ts,
                action=str(record.get("action", "")),
                expected_direction=str(record.get("expected_direction", "")),
                confidence=float(record.get("confidence", 0.0)),
                executed_price_cny=record.get("executed_price_cny"),
                shares=record.get("shares"),
                fees_cny=record.get("fees_cny"),
                realized_pnl_cny=record.get("realized_pnl_cny"),
                comment=record.get("comment"),
            )

    async def clear(self, symbol: str) -> int:
        async with session_scope(self.session_factory) as s:
            repo = TradeFeedbackRepo(s)
            return await repo.clear_symbol(symbol)
