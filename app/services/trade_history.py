# app/services/trade_history.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


@dataclass
class TradeHistoryStore:
    """
    Minimal in-memory store.
    Replace with DB later (SQLite/Postgres) without changing StrategyEngine.
    """
    records: List[Dict[str, Any]] = field(default_factory=list)
    max_records: int = 200

    def list_records(self) -> List[Dict[str, Any]]:
        return list(self.records)

    def append(self, record: Dict[str, Any]) -> None:
        self.records.append(record)
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records :]

    def append_from_trade_result(
        self,
        *,
        decision_ts: str,
        action: str,
        expected_direction: str,
        confidence: float,
        executed_price_cny: Optional[float],
        shares: Optional[int],
        fees_cny: Optional[float],
        realized_pnl_cny: Optional[float],
        comment: str,
    ) -> None:
        self.append(
            {
                "decision_ts": decision_ts,
                "action": action,
                "expected_direction": expected_direction,
                "confidence": float(confidence),
                "executed_price_cny": executed_price_cny,
                "shares": shares,
                "fees_cny": fees_cny,
                "realized_pnl_cny": realized_pnl_cny,
                "comment": comment,
            }
        )


def utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
