# app/services/strategy.py
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Optional, Tuple, Any, Dict, List

from app.logging_config import logger
from app.models import (
    TradeSignal,
    TradeRequest,
    TradeResult,
    MarketSnapshot,
    AccountState,
    ExecutionConstraints,
)
from app.services.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from app.services.risk import RiskManager
from app.services.broker import Broker
from app.services.market_data import MarketDataProvider
from app.services.llm_client import OpenAICompatLLM
from app.services.trade_history_db import TradeHistoryDB


SIGNAL_SCHEMA_HINT = {
    "type": "object",
    "required": [
        "action",
        "horizon_minutes",
        "confidence",
        "reason",
        "suggested_lots",
        "expected_direction",
        "risk_notes",
    ],
    "properties": {
        "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "horizon_minutes": {"type": "integer", "enum": [30]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
        "suggested_lots": {"type": "integer", "minimum": 0},
        "expected_direction": {"type": "string", "enum": ["UP", "DOWN", "FLAT"]},
        "risk_notes": {"type": "string"},
    },
    "additionalProperties": False,
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _jsonable(obj: Any) -> Any:
    """Convert pydantic models / datetimes into json-safe structures."""
    if obj is None:
        return None
    # pydantic v2 models
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    # plain datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

class StrategyEngine:
    """
    Provider-agnostic strategy engine.

    Flow:
      snapshot (market) + intraday_records_today (db) + account + constraints
        -> prompt_builder
        -> LLM JSON TradeSignal
        -> risk gating
        -> broker place_order (paper)
      All decisions are persisted as "intraday records" for feedback-loop.

    Requires TradeHistoryDB to implement:
      - list_intraday_today(symbol, now_ts, limit) -> List[Dict]
      - append_intraday(symbol, now_ts, record) -> None
    """

    def __init__(
        self,
        market: MarketDataProvider,
        llm: OpenAICompatLLM,
        risk: RiskManager,
        broker: Broker,
        *,
        fixed_lots: int = 1,
        trade_history_db: TradeHistoryDB,
        timezone_name: str = "Asia/Shanghai",
        lot_size: int = 100,
        max_order_shares: Optional[int] = 10000,
        fees_bps_est: int = 5,
        slippage_bps_est: int = 5,
        trade_history_limit: int = 10,
        market_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.market = market
        self.llm = llm
        self.risk = risk
        self.broker = broker

        self.fixed_lots = int(fixed_lots)
        self.trade_history_db = trade_history_db
        self.trade_history_limit = int(trade_history_limit)

        self.timezone_name = timezone_name
        self.lot_size = int(lot_size)
        self.max_order_shares = max_order_shares
        self.fees_bps_est = int(fees_bps_est)
        self.slippage_bps_est = int(slippage_bps_est)

        self.market_kwargs = dict(market_kwargs or {})

        logger.info(
            "StrategyEngine initialized | lot_size=%s max_order_shares=%s fixed_lots=%s"
            "min_confidence=%s trade_history_limit=%s market_kwargs=%s",
            self.lot_size,
            self.max_order_shares,
            self.fixed_lots,
            getattr(self.risk, "min_confidence", None),
            self.trade_history_limit,
            self.market_kwargs,
        )

    async def _snapshot_and_signal(
        self,
        symbol: str,
        *,
        account: AccountState,
        now_ts: Optional[str] = None,
    ) -> Tuple[MarketSnapshot, ExecutionConstraints, TradeSignal, List[Dict[str, Any]]]:
        t0 = time.perf_counter()
        sym = symbol.strip()
        trades_left = self.risk.trades_left()

        logger.info(
            "signal_start | symbol=%s cash_cny=%.2f pos=%s now_ts=%s trades_left=%s",
            sym,
            float(account.cash_cny),
            int(account.position_shares),
            now_ts,
            trades_left,
        )

        # 1) snapshot
        snap_t0 = time.perf_counter()
        snapshot = await self.market.get_snapshot(sym, end_ts=now_ts, **self.market_kwargs)
        snap_ms = (time.perf_counter() - snap_t0) * 1000.0

        # 2) constraints
        constraints = ExecutionConstraints(
            horizon_minutes=30,
            max_trades_left_today=trades_left,
            lot_size=self.lot_size,
            max_order_shares=self.max_order_shares,
            min_confidence_to_trade=self.risk.min_confidence,
            fees_bps_est=self.fees_bps_est,
            slippage_bps_est=self.slippage_bps_est,
        )

        # 3) intraday trade records (Shanghai today)
        hist_t0 = time.perf_counter()
        try:
            intraday_records_today = await self.trade_history_db.list_intraday_today(
                symbol=sym,
                now_ts=now_ts,
                limit=self.trade_history_limit,
            )
        except Exception as e:
            logger.warning("trade_history_list_failed | symbol=%s err=%s: %s", sym, type(e).__name__, e)
            intraday_records_today = []
        hist_ms = (time.perf_counter() - hist_t0) * 1000.0

        # 4) prompt
        prompt = build_user_prompt(
            snapshot,
            now_ts=now_ts or (snapshot.ts.strftime("%Y-%m-%d %H:%M:%S") if snapshot.ts else None),
            timezone_name=self.timezone_name,
            account=account,
            constraints=constraints,
            trade_history_records=intraday_records_today,
        )

        # 5) LLM
        llm_t0 = time.perf_counter()
        signal: TradeSignal = await self.llm.chat_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            schema_hint=SIGNAL_SCHEMA_HINT,
        )
        llm_ms = (time.perf_counter() - llm_t0) * 1000.0

        # enforce horizon
        signal.horizon_minutes = 30

        # normalize lots
        if signal.action in ("BUY", "SELL"):
            if int(signal.suggested_lots) <= 0 and self.fixed_lots > 0:
                signal.suggested_lots = self.fixed_lots

            if self.max_order_shares is not None and self.max_order_shares > 0:
                max_lots_by_cap = max(int(self.max_order_shares // max(self.lot_size, 1)), 0)
                signal.suggested_lots = min(int(signal.suggested_lots), max_lots_by_cap)
        else:
            signal.suggested_lots = 0

        shares_hint = int(signal.suggested_lots) * int(constraints.lot_size)

        total_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "signal_done | symbol=%s action=%s conf=%.3f dir=%s lots=%s shares=%s "
            "bars=%s hist=%s ms_total=%.1f ms_snap=%.1f ms_hist=%.1f ms_llm=%.1f last_price=%s",
            sym,
            signal.action,
            float(signal.confidence),
            signal.expected_direction,
            int(signal.suggested_lots),
            shares_hint,
            len(snapshot.recent_bars or []),
            len(intraday_records_today),
            total_ms,
            snap_ms,
            hist_ms,
            llm_ms,
            getattr(snapshot, "last_price", None),
        )

        return snapshot, constraints, signal, intraday_records_today

    async def get_signal(self, symbol: str, *, account: AccountState, now_ts: Optional[str] = None) -> TradeSignal:
        _, _, signal, _ = await self._snapshot_and_signal(symbol, account=account, now_ts=now_ts)
        return signal

    async def maybe_execute(self, symbol: str, *, account: AccountState, now_ts: Optional[str] = None) -> TradeResult:
        t0 = time.perf_counter()
        sym = symbol.strip()

        snapshot, constraints, signal, _ = await self._snapshot_and_signal(sym, account=account, now_ts=now_ts)

        suggested_lots = int(signal.suggested_lots) if signal.action in ("BUY", "SELL") else 0
        suggested_shares = suggested_lots * int(constraints.lot_size)

        async def _save_intraday(
            *,
            status: str,
            ok: bool,
            message: str,
            broker_details: Optional[Dict[str, Any]] = None,
        ) -> None:
            try:
                decision_dt = snapshot.ts or utcnow()
                record = {
                    "decision_ts": decision_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "action": signal.action,
                    "expected_direction": signal.expected_direction,
                    "confidence": float(signal.confidence),
                    "reason": signal.reason,
                    "risk_notes": signal.risk_notes,
                    "lot_size": int(constraints.lot_size),
                    "suggested_lots": int(suggested_lots),
                    "suggested_shares": int(suggested_shares),
                    "status": status,
                    "ok": bool(ok),
                    "message": message,
                    "signal_json": json.dumps(_jsonable(signal), ensure_ascii=False),
                    "snapshot_json": json.dumps(_jsonable(snapshot), ensure_ascii=False),
                    "broker_details_json": json.dumps(_jsonable(broker_details), ensure_ascii=False) if broker_details else None,
                    "executed_price_cny": None,
                    "fees_cny": None,
                    "realized_pnl_cny": None,
                }
                await self.trade_history_db.append_intraday(symbol=sym, now_ts=now_ts, record=record)
            except Exception as e:
                logger.warning("intraday_save_failed | symbol=%s err=%s: %s", sym, type(e).__name__, e)

        # HOLD
        if signal.action == "HOLD":
            logger.info("execute_skip_hold | symbol=%s conf=%.3f reason=%r", sym, float(signal.confidence), signal.reason)
            await _save_intraday(status="HOLD", ok=True, message="hold (no order placed)")
            return TradeResult(
                ok=True,
                message="hold (no order placed)",
                ts=snapshot.ts or utcnow(),
                symbol=sym,
                action="HOLD",
                shares=0,
                lots=0,
                paper=True,
                details={"signal": signal.model_dump(), "snapshot": snapshot.model_dump()},
            )

        # risk gating (NOTE: your RiskManager must accept suggested_shares + lot_size)
        ok, why = self.risk.can_trade(
            signal,
            account=account,
            snapshot=snapshot,
            suggested_shares=int(suggested_shares),
            lot_size=int(constraints.lot_size),
        )

        if not ok:
            logger.warning(
                "execute_blocked | symbol=%s action=%s conf=%.3f why=%s",
                sym,
                signal.action,
                float(signal.confidence),
                why,
            )
            await _save_intraday(status="BLOCKED", ok=False, message=f"blocked: {why}")
            return TradeResult(
                ok=False,
                message=f"blocked: {why}",
                ts=snapshot.ts or utcnow(),
                symbol=sym,
                action=signal.action,
                shares=int(suggested_shares),
                lots=int(suggested_lots),
                paper=True,
                details={"signal": signal.model_dump(), "snapshot": snapshot.model_dump()},
            )

        # broker order (shares-based)
        req = TradeRequest(
            symbol=sym,
            action=signal.action,
            shares=int(suggested_shares),
            lot_size=int(constraints.lot_size),
        )

        broker_t0 = time.perf_counter()
        res = await self.broker.place_order(req)
        broker_ms = (time.perf_counter() - broker_t0) * 1000.0

        # attach for API return/debug
        res.details.setdefault("signal", signal.model_dump())
        res.details.setdefault("snapshot", snapshot.model_dump())

        logger.info(
            "execute_done | symbol=%s ok=%s action=%s lots=%s shares=%s broker_ms=%.1f msg=%r",
            sym,
            bool(res.ok),
            res.action,
            int(suggested_lots),
            int(suggested_shares),
            broker_ms,
            res.message,
        )

        await _save_intraday(
            status="EXECUTED" if res.ok else "FAILED",
            ok=bool(res.ok),
            message=str(res.message),
            broker_details=res.details,
        )

        if res.ok:
            self.risk.record_trade()

        total_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("execute_total | symbol=%s ms_total=%.1f", sym, total_ms)
        return res
