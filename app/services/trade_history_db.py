# app/services/trade_history_db.py
from __future__ import annotations

import json
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo

from app.storage.db import session_scope
from app.storage.repo import IntradayTradeRepo

CN_TZ = ZoneInfo("Asia/Shanghai")


def _parse_sh_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.strptime(str(s).strip(), "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=CN_TZ)
    except Exception:
        return None


def _trading_day_sh(now_ts: Optional[str], fallback_dt: Optional[datetime] = None) -> date:
    dt = _parse_sh_ts(now_ts)
    if dt:
        return dt.date()
    if fallback_dt:
        if fallback_dt.tzinfo is None:
            # assume UTC if naive
            fallback_dt = fallback_dt.replace(tzinfo=ZoneInfo("UTC"))
        return fallback_dt.astimezone(CN_TZ).date()
    return datetime.now(CN_TZ).date()


class TradeHistoryDB:
    """
    DB-backed intraday trade history used by prompt_builder.
    Stores every decision/execution attempt and provides "today intraday" slices.
    """

    def __init__(self, session_factory, *, max_records: int = 50):
        self.session_factory = session_factory
        self.max_records = int(max_records)

    async def list_intraday_today(
        self,
        symbol: str,
        *,
        now_ts: Optional[str],
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return Shanghai-today records (chronological) for LLM prompt usage.
        """
        limit_n = min(int(limit or self.max_records), self.max_records)

        async with session_scope(self.session_factory) as s:
            repo = IntradayTradeRepo(s)
            day = _trading_day_sh(now_ts)
            rows = await repo.list_by_day(symbol, day, limit=limit_n, asc=True)

        # convert ORM -> prompt-friendly dict
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "decision_ts": r.decision_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "action": r.action,
                    "expected_direction": r.expected_direction,
                    "confidence": r.confidence,
                    "suggested_lots": r.suggested_lots,
                    "suggested_shares": r.suggested_shares,
                    "fees_cny": r.fees_cny,
                    "realized_pnl_cny": r.realized_pnl_cny,
                    "status": r.status,
                    "message": r.message,
                }
            )
        return out

    async def append_intraday(
        self,
        symbol: str,
        *,
        now_ts: Optional[str],
        record: Dict[str, Any],
    ) -> None:
        """
        Persist a decision/execution attempt row.

        `record` should already contain:
          - decision_ts (datetime or "YYYY-MM-DD HH:MM:SS")
          - action, expected_direction, confidence, reason, risk_notes
          - lot_size, suggested_lots, suggested_shares
          - status, ok, message
          - signal_json, snapshot_json, broker_details_json (optional)
          - executed_price_cny, fees_cny, realized_pnl_cny (optional)
        """
        decision_ts = record.get("decision_ts")
        if isinstance(decision_ts, str):
            # interpret string as Shanghai local time (consistent with your prompts/logs)
            dt_sh = _parse_sh_ts(decision_ts)
            if dt_sh is None:
                return
            # store as naive or aware; pick one. I recommend storing naive UTC or aware dt.
            # We'll store naive in SQLite for simplicity by dropping tzinfo:
            decision_dt = dt_sh.replace(tzinfo=None)
        elif isinstance(decision_ts, datetime):
            decision_dt = decision_ts.replace(tzinfo=None)
        else:
            return

        day = _trading_day_sh(now_ts, fallback_dt=decision_dt)

        row = {
            "symbol": symbol,
            "trading_day_sh": day,
            "decision_ts": decision_dt,
            "action": str(record.get("action", "")),
            "expected_direction": record.get("expected_direction"),
            "confidence": record.get("confidence"),
            "reason": record.get("reason"),
            "risk_notes": record.get("risk_notes"),
            "lot_size": int(record.get("lot_size") or 100),
            "suggested_lots": int(record.get("suggested_lots") or 0),
            "suggested_shares": int(record.get("suggested_shares") or 0),
            "status": str(record.get("status") or "DECIDED"),
            "ok": 1 if bool(record.get("ok")) else 0,
            "message": record.get("message"),
            "executed_price_cny": record.get("executed_price_cny"),
            "fees_cny": record.get("fees_cny"),
            "realized_pnl_cny": record.get("realized_pnl_cny"),
            "signal_json": record.get("signal_json"),
            "snapshot_json": record.get("snapshot_json"),
            "broker_details_json": record.get("broker_details_json"),
        }

        async with session_scope(self.session_factory) as s:
            repo = IntradayTradeRepo(s)
            await repo.add(row)

    async def clear(self, symbol: str) -> int:
        async with session_scope(self.session_factory) as s:
            repo = IntradayTradeRepo(s)
            return await repo.clear_symbol(symbol)
