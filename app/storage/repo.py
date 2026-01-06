# app/storage/repo.py
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select, delete, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage.orm_models import TradeFeedbackORM, ExecutionORM


class TradeFeedbackRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_feedback(
        self,
        *,
        symbol: str,
        decision_ts: datetime,
        action: str,
        expected_direction: str,
        confidence: float,
        executed_price_cny: Optional[float] = None,
        shares: Optional[int] = None,
        fees_cny: Optional[float] = None,
        realized_pnl_cny: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> TradeFeedbackORM:
        row = TradeFeedbackORM(
            symbol=symbol,
            decision_ts=decision_ts,
            action=action,
            expected_direction=expected_direction,
            confidence=confidence,
            executed_price_cny=executed_price_cny,
            shares=shares,
            fees_cny=fees_cny,
            realized_pnl_cny=realized_pnl_cny,
            comment=comment,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def list_recent(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        q = (
            select(TradeFeedbackORM)
            .where(TradeFeedbackORM.symbol == symbol)
            .order_by(desc(TradeFeedbackORM.decision_ts))
            .limit(limit)
        )
        rows = (await self.session.execute(q)).scalars().all()
        # Convert to dict compatible with your prompt "trade_history.records"
        return [
            {
                "decision_ts": r.decision_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "action": r.action,
                "expected_direction": r.expected_direction,
                "confidence": r.confidence,
                "executed_price_cny": r.executed_price_cny,
                "shares": r.shares,
                "fees_cny": r.fees_cny,
                "realized_pnl_cny": r.realized_pnl_cny,
                "comment": r.comment,
            }
            for r in rows
        ][::-1]  # return chronological ascending


    async def clear_symbol(self, symbol: str) -> int:
        res = await self.session.execute(delete(TradeFeedbackORM).where(TradeFeedbackORM.symbol == symbol))
        return int(res.rowcount or 0)


class ExecutionRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_execution(
        self,
        *,
        symbol: str,
        ts: datetime,
        action: str,
        notional_usd: float,
        paper: bool,
        ok: bool,
        message: str | None,
        signal: Dict[str, Any] | None,
        snapshot: Dict[str, Any] | None,
        broker_details: Dict[str, Any] | None,
    ) -> ExecutionORM:
        row = ExecutionORM(
            symbol=symbol,
            ts=ts,
            action=action,
            notional_usd=float(notional_usd or 0.0),
            paper=1 if paper else 0,
            ok=1 if ok else 0,
            message=message,
            signal_json=json.dumps(signal, ensure_ascii=False) if signal else None,
            snapshot_json=json.dumps(snapshot, ensure_ascii=False) if snapshot else None,
            broker_details_json=json.dumps(broker_details, ensure_ascii=False) if broker_details else None,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def list_recent(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        q = (
            select(ExecutionORM)
            .where(ExecutionORM.symbol == symbol)
            .order_by(desc(ExecutionORM.ts))
            .limit(limit)
        )
        rows = (await self.session.execute(q)).scalars().all()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "ts": r.ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "action": r.action,
                    "notional_usd": r.notional_usd,
                    "paper": bool(r.paper),
                    "ok": bool(r.ok),
                    "message": r.message,
                }
            )
        return out[::-1]
