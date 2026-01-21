# app/services/strategy.py
from __future__ import annotations

import json
import time
from datetime import datetime
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
from app.utils.timeutils import now_shanghai, ensure_shanghai, fmt_shanghai


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


def sh_now() -> datetime:
    """Shanghai tz-aware now."""
    return now_shanghai()


def _jsonable(obj: Any) -> Any:
    """Convert pydantic models / datetimes into json-safe structures."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, datetime):
        return ensure_shanghai(obj).isoformat()
    return obj


def _safe_json_dumps(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    try:
        return json.dumps(_jsonable(obj), ensure_ascii=False)
    except Exception:
        try:
            return json.dumps(str(obj), ensure_ascii=False)
        except Exception:
            return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if (v == v) else None  # NaN guard
    except Exception:
        return None


def _estimate_realized_pnl_cny_roundtrip(
    *,
    sell_price_cny: float,
    sell_shares: int,
    account: AccountState,
    sell_total_cost_cny: float,
) -> Optional[float]:
    """
    Best-effort realized PnL estimate for SELL.

    Assumptions:
      - account.avg_cost_cny is average cost per share EXCLUDING fees (common).
      - We subtract SELL-side total friction cost passed in (sell_total_cost_cny).
      - BUY-side fees are NOT precisely known here (paper broker doesn't update avg_cost),
        so we do NOT allocate buy-fees again to avoid double counting.

    realized_pnl ~= (sell_price - avg_cost) * shares - sell_total_cost
    """
    if sell_price_cny <= 0 or sell_shares <= 0:
        return None

    avg_cost = _safe_float(account.avg_cost_cny)
    if avg_cost is None or avg_cost <= 0:
        return None

    gross = (float(sell_price_cny) - float(avg_cost)) * int(sell_shares)
    net = gross - float(sell_total_cost_cny or 0.0)
    return float(net)


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
        trade_history_db: TradeHistoryDB,
        fixed_lots: int = 1,
        trade_history_limit: int = 10,
    ):
        self.market = market
        self.llm = llm
        self.risk = risk
        self.broker = broker
        self.trade_history_db = trade_history_db

        self.fixed_lots = int(fixed_lots)
        self.trade_history_limit = int(trade_history_limit)

        logger.info(
            "StrategyEngine initialized | fixed_lots=%s min_confidence=%s trade_history_limit=%s",
            self.fixed_lots,
            getattr(self.risk, "min_confidence", None),
            self.trade_history_limit,
        )

    @staticmethod
    def _normalize_market_kwargs(market_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return dict(market_kwargs or {})

    @staticmethod
    def _market_contract_hints(market_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        kw = dict(market_kwargs or {})
        min_period = kw.get("min_period") or kw.get("period") or kw.get("bar_freq_minutes")
        lookback = kw.get("min_lookback_minutes") or kw.get("lookback_minutes")
        return {
            "min_period": str(min_period) if min_period is not None else None,
            "min_lookback_minutes": int(lookback) if lookback is not None else None,
        }

    @staticmethod
    def _compute_suggested_shares(
        signal: TradeSignal,
        *,
        lot_size: int,
        fixed_lots: int,
        max_order_shares: Optional[int],
    ) -> Tuple[int, int]:
        """
        Returns (suggested_lots, suggested_shares) after normalization/clamping.
        """
        if signal.action not in ("BUY", "SELL"):
            return 0, 0

        lots = int(getattr(signal, "suggested_lots", 0) or 0)
        if lots <= 0 and fixed_lots > 0:
            lots = int(fixed_lots)

        ls = max(int(lot_size), 1)

        if max_order_shares is not None and int(max_order_shares) > 0:
            max_lots_by_cap = max(int(int(max_order_shares) // ls), 0)
            lots = min(int(lots), max_lots_by_cap)

        shares = int(lots) * ls
        return int(lots), int(shares)

    def _decision_ts_shanghai(self, snapshot: MarketSnapshot) -> Tuple[datetime, str]:
        """
        Returns (decision_dt_tzaware, decision_ts_sh_str).
        decision_ts string uses "YYYY-MM-DD HH:MM:SS" in Asia/Shanghai.
        """
        decision_dt = ensure_shanghai(snapshot.ts) if getattr(snapshot, "ts", None) else sh_now()
        decision_ts_sh = fmt_shanghai(decision_dt) or ""
        return decision_dt, decision_ts_sh

    async def _persist_intraday_record(
        self,
        *,
        symbol: str,
        now_ts_sh: str,
        signal: TradeSignal,
        snapshot: MarketSnapshot,
        constraints: ExecutionConstraints,
        suggested_lots: int,
        suggested_shares: int,
        status: str,
        ok: bool,
        message: str,
        broker_details: Optional[Dict[str, Any]] = None,
        trade_result: Optional[TradeResult] = None,
        realized_pnl_cny: Optional[float] = None,
    ) -> None:
        """
        Persist an intraday record.

        ✅ Requirement:
          executed_price_cny := snapshot.last_price (last stock price), for all statuses.

        ✅ Option A:
          fees_cny := total_cost_cny (all-in friction) for backward compatibility.
          fee_cny/slippage_cny/total_cost_cny are the detailed breakdown.
        """
        try:
            last_px = _safe_float(getattr(snapshot, "last_price", None))

            # Pull cost fields from TradeResult if provided
            tr = trade_result
            executed_price_net_cny = _safe_float(getattr(tr, "executed_price_net_cny", None)) if tr else None
            notional_cny = _safe_float(getattr(tr, "notional_cny", None)) if tr else None
            fee_cny = _safe_float(getattr(tr, "fee_cny", None)) if tr else None
            slippage_cny = _safe_float(getattr(tr, "slippage_cny", None)) if tr else None
            total_cost_cny = _safe_float(getattr(tr, "total_cost_cny", None)) if tr else None
            cash_delta_cny = _safe_float(getattr(tr, "cash_delta_cny", None)) if tr else None

            # ✅ Option A enforcement: legacy fees_cny mirrors total_cost_cny
            fees_cny = total_cost_cny

            record: Dict[str, Any] = {
                "decision_ts": now_ts_sh,
                "action": getattr(signal, "action", ""),
                "expected_direction": getattr(signal, "expected_direction", None),
                "confidence": float(getattr(signal, "confidence", 0.0) or 0.0),
                "reason": getattr(signal, "reason", None),
                "risk_notes": getattr(signal, "risk_notes", None),
                "lot_size": int(getattr(constraints, "lot_size", 100) or 100),
                "suggested_lots": int(suggested_lots),
                "suggested_shares": int(suggested_shares),
                "status": str(status),
                "ok": bool(ok),
                "message": str(message),
                "signal_json": _safe_json_dumps(signal),
                "snapshot_json": _safe_json_dumps(snapshot),
                "broker_details_json": _safe_json_dumps(broker_details) if broker_details else None,

                # ✅ required: always store snapshot.last_price here
                "executed_price_cny": last_px,

                # ✅ Option A + detailed costs
                "fees_cny": fees_cny,
                "executed_price_net_cny": executed_price_net_cny,
                "notional_cny": notional_cny,
                "fee_cny": fee_cny,
                "slippage_cny": slippage_cny,
                "total_cost_cny": total_cost_cny,
                "cash_delta_cny": cash_delta_cny,

                "realized_pnl_cny": _safe_float(realized_pnl_cny),
            }

            await self.trade_history_db.append_intraday(symbol=symbol, now_ts=now_ts_sh, record=record)
        except Exception as e:
            logger.warning("intraday_save_failed | symbol=%s err=%s: %s", symbol, type(e).__name__, e)

    async def _snapshot_and_signal(
        self,
        symbol: str,
        *,
        account: AccountState,
        now_ts: Optional[str] = None,
        timezone_name: str = "Asia/Shanghai",
        lot_size: int = 100,
        max_order_shares: Optional[int] = 10000,
        fees_bps_est: int = 5,
        slippage_bps_est: int = 5,
        market_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MarketSnapshot, ExecutionConstraints, TradeSignal, List[Dict[str, Any]]]:
        t0 = time.perf_counter()
        sym = symbol.strip()

        market_kwargs_n = self._normalize_market_kwargs(market_kwargs)
        trades_left = self.risk.trades_left()

        logger.info(
            "signal_start | symbol=%s cash_cny=%.2f pos=%s now_ts=%s trades_left=%s "
            "tz=%s lot_size=%s max_order_shares=%s fees_bps=%s slip_bps=%s market_kwargs=%s",
            sym,
            float(account.cash_cny),
            int(account.position_shares),
            now_ts,
            trades_left,
            timezone_name,
            int(lot_size),
            max_order_shares,
            int(fees_bps_est),
            int(slippage_bps_est),
            market_kwargs_n,
        )

        # 1) snapshot
        snap_t0 = time.perf_counter()
        snapshot = await self.market.get_snapshot(sym, end_ts=now_ts, **market_kwargs_n)
        snapshot.ts = ensure_shanghai(snapshot.ts)
        snap_ms = (time.perf_counter() - snap_t0) * 1000.0

        # 2) constraints
        contract_hints = self._market_contract_hints(market_kwargs_n)
        constraints = ExecutionConstraints(
            horizon_minutes=30,
            max_trades_left_today=trades_left,
            lot_size=int(lot_size),
            max_order_shares=max_order_shares,
            min_confidence_to_trade=self.risk.min_confidence,
            fees_bps_est=int(fees_bps_est),
            slippage_bps_est=int(slippage_bps_est),
            market_contract=contract_hints,
        )

        # If caller didn't provide now_ts, anchor to snapshot.ts
        now_ts_sh = now_ts or (fmt_shanghai(snapshot.ts) if snapshot.ts else None)

        # 3) intraday trade records
        hist_t0 = time.perf_counter()
        try:
            intraday_records_today = await self.trade_history_db.list_intraday_today(
                symbol=sym,
                now_ts=now_ts_sh,
                limit=self.trade_history_limit,
            )
        except Exception as e:
            logger.warning("trade_history_list_failed | symbol=%s err=%s: %s", sym, type(e).__name__, e)
            intraday_records_today = []
        hist_ms = (time.perf_counter() - hist_t0) * 1000.0

        # 4) prompt
        prompt = build_user_prompt(
            snapshot,
            now_ts=now_ts_sh,
            timezone_name=timezone_name,
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

        signal.horizon_minutes = 30

        # normalize lots/shares
        suggested_lots, suggested_shares = self._compute_suggested_shares(
            signal,
            lot_size=int(constraints.lot_size),
            fixed_lots=self.fixed_lots,
            max_order_shares=max_order_shares,
        )
        signal.suggested_lots = suggested_lots

        total_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "signal_done | symbol=%s action=%s conf=%.3f dir=%s lots=%s shares=%s "
            "bars=%s hist=%s ms_total=%.1f ms_snap=%.1f ms_hist=%.1f ms_llm=%.1f last_price=%s now_ts=%s contract=%s",
            sym,
            signal.action,
            float(signal.confidence),
            signal.expected_direction,
            int(suggested_lots),
            int(suggested_shares),
            len(snapshot.recent_bars or []),
            len(intraday_records_today),
            total_ms,
            snap_ms,
            hist_ms,
            llm_ms,
            getattr(snapshot, "last_price", None),
            now_ts_sh,
            constraints.market_contract,
        )

        return snapshot, constraints, signal, intraday_records_today

    async def get_signal(
        self,
        symbol: str,
        *,
        account: AccountState,
        now_ts: Optional[str] = None,
        timezone_name: str = "Asia/Shanghai",
        lot_size: int = 100,
        max_order_shares: Optional[int] = 10000,
        fees_bps_est: int = 5,
        slippage_bps_est: int = 5,
        market_kwargs: Optional[Dict[str, Any]] = None,
    ) -> TradeSignal:
        _, _, signal, _ = await self._snapshot_and_signal(
            symbol,
            account=account,
            now_ts=now_ts,
            timezone_name=timezone_name,
            lot_size=lot_size,
            max_order_shares=max_order_shares,
            fees_bps_est=fees_bps_est,
            slippage_bps_est=slippage_bps_est,
            market_kwargs=market_kwargs,
        )
        return signal

    async def maybe_execute(
        self,
        symbol: str,
        *,
        account: AccountState,
        now_ts: Optional[str] = None,
        timezone_name: str = "Asia/Shanghai",
        lot_size: int = 100,
        max_order_shares: Optional[int] = 10000,
        fees_bps_est: int = 5,
        slippage_bps_est: int = 5,
        market_kwargs: Optional[Dict[str, Any]] = None,
    ) -> TradeResult:
        t0 = time.perf_counter()
        sym = symbol.strip()

        snapshot, constraints, signal, _ = await self._snapshot_and_signal(
            sym,
            account=account,
            now_ts=now_ts,
            timezone_name=timezone_name,
            lot_size=lot_size,
            max_order_shares=max_order_shares,
            fees_bps_est=fees_bps_est,
            slippage_bps_est=slippage_bps_est,
            market_kwargs=market_kwargs,
        )

        suggested_lots, suggested_shares = self._compute_suggested_shares(
            signal,
            lot_size=int(constraints.lot_size),
            fixed_lots=self.fixed_lots,
            max_order_shares=max_order_shares,
        )

        decision_dt, decision_ts_sh = self._decision_ts_shanghai(snapshot)

        # HOLD
        if signal.action == "HOLD":
            logger.info(
                "execute_skip_hold | symbol=%s conf=%.3f reason=%r",
                sym,
                float(signal.confidence),
                signal.reason,
            )
            await self._persist_intraday_record(
                symbol=sym,
                now_ts_sh=decision_ts_sh,
                signal=signal,
                snapshot=snapshot,
                constraints=constraints,
                suggested_lots=0,
                suggested_shares=0,
                status="HOLD",
                ok=True,
                message="hold (no order placed)",
            )
            return TradeResult(
                ok=True,
                message="hold (no order placed)",
                ts=decision_dt,
                symbol=sym,
                action="HOLD",
                shares=0,
                lots=0,
                paper=True,
                details={"signal": signal.model_dump(), "snapshot": snapshot.model_dump()},
            )

        # risk gating
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
            await self._persist_intraday_record(
                symbol=sym,
                now_ts_sh=decision_ts_sh,
                signal=signal,
                snapshot=snapshot,
                constraints=constraints,
                suggested_lots=int(suggested_lots),
                suggested_shares=int(suggested_shares),
                status="BLOCKED",
                ok=False,
                message=f"blocked: {why}",
            )
            return TradeResult(
                ok=False,
                message=f"blocked: {why}",
                ts=decision_dt,
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
            price_cny=_safe_float(getattr(snapshot, "last_price", None)),  # ✅ reference price for cost model
        )

        broker_t0 = time.perf_counter()
        res = await self.broker.place_order(req)
        broker_ms = (time.perf_counter() - broker_t0) * 1000.0

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

        # Best-effort realized pnl estimate (SELL only)
        realized_pnl_cny: Optional[float] = None
        if res.ok and str(signal.action) == "SELL":
            last_px = _safe_float(getattr(snapshot, "last_price", None)) or 0.0
            realized_pnl_cny = _estimate_realized_pnl_cny_roundtrip(
                sell_price_cny=float(last_px),
                sell_shares=int(suggested_shares),
                account=account,
                sell_total_cost_cny=float(res.total_cost_cny or 0.0),  # Option A friction
            )
        elif res.ok and str(signal.action) == "BUY":
            realized_pnl_cny = 0.0

        # Optional: expose in details for UI/debug
        if res.ok:
            try:
                res.details.setdefault("costs", {})
                res.details["costs"].update(
                    {
                        "executed_price_cny": _safe_float(res.executed_price_cny),
                        "executed_price_net_cny": _safe_float(res.executed_price_net_cny),
                        "fee_cny": _safe_float(res.fee_cny),
                        "slippage_cny": _safe_float(res.slippage_cny),
                        "total_cost_cny": _safe_float(res.total_cost_cny),
                        "fees_cny_legacy": _safe_float(res.total_cost_cny),  # fees_cny == total_cost_cny
                        "cash_delta_cny": _safe_float(res.cash_delta_cny),
                    }
                )
            except Exception:
                pass

        # Save intraday record (executed_price_cny := snapshot.last_price)
        await self._persist_intraday_record(
            symbol=sym,
            now_ts_sh=decision_ts_sh,
            signal=signal,
            snapshot=snapshot,
            constraints=constraints,
            suggested_lots=int(suggested_lots),
            suggested_shares=int(suggested_shares),
            status="EXECUTED" if res.ok else "FAILED",
            ok=bool(res.ok),
            message=str(res.message),
            broker_details=res.details,
            trade_result=res if res.ok else None,  # persist cost fields only when ok
            realized_pnl_cny=realized_pnl_cny,
        )

        if res.ok:
            self.risk.record_trade()

        total_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("execute_total | symbol=%s ms_total=%.1f", sym, total_ms)
        return res
