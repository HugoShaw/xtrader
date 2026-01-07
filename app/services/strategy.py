# app/services/strategy.py
from __future__ import annotations

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
        "suggested_notional_cny",
        "expected_direction",
        "risk_notes",
    ],
    "properties": {
        "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "horizon_minutes": {"type": "integer", "enum": [30]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
        "suggested_notional_cny": {"type": "number", "minimum": 0.0},
        "expected_direction": {"type": "string", "enum": ["UP", "DOWN", "FLAT"]},
        "risk_notes": {"type": "string"},
    },
    "additionalProperties": False,
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StrategyEngine:
    """
    Provider-agnostic strategy engine.

    Works with AkShareAStockProvider or any provider implementing MarketDataProvider.get_snapshot().
    """

    def __init__(
        self,
        market: MarketDataProvider,
        llm: OpenAICompatLLM,
        risk: RiskManager,
        broker: Broker,
        *,
        fixed_amount_cny: float,
        trade_history_db: TradeHistoryDB,
        timezone_name: str = "Asia/Shanghai",
        lot_size: int = 50, # 几手
        max_order_shares: Optional[int] = 10000,
        fees_bps_est: int = 5,
        slippage_bps_est: int = 5,
        trade_history_limit: int = 50,
        # ✅ optional: passed to market.get_snapshot() (works nicely with AkShare provider knobs)
        market_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.market = market
        self.llm = llm
        self.risk = risk
        self.broker = broker

        # treat as a "cap" or "default" sizing hint; disable by setting <=0
        self.fixed_amount_cny = float(fixed_amount_cny)

        self.trade_history_db = trade_history_db
        self.trade_history_limit = int(trade_history_limit)

        self.timezone_name = timezone_name
        self.lot_size = int(lot_size)
        self.max_order_shares = max_order_shares
        self.fees_bps_est = int(fees_bps_est)
        self.slippage_bps_est = int(slippage_bps_est)

        self.market_kwargs = dict(market_kwargs or {})

        logger.info(
            "StrategyEngine initialized | lot_size=%s max_order_shares=%s fixed_amount_cny=%s "
            "min_confidence=%s trade_history_limit=%s market_kwargs=%s",
            self.lot_size,
            self.max_order_shares,
            self.fixed_amount_cny,
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
    ) -> Tuple[MarketSnapshot, TradeSignal]:
        """
        Fetch snapshot -> load recent trade history -> build prompt -> LLM JSON output.
        """
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

        # 1) market snapshot (provider-specific kwargs are optional)
        snap_t0 = time.perf_counter()
        end_ts = None
        if now_ts:
            end_ts = now_ts  # keep as string or parse to datetime once

        snapshot = await self.market.get_snapshot(sym, end_ts=end_ts, **self.market_kwargs) 
        snap_ms = (time.perf_counter() - snap_t0) * 1000.0

        # ---- DEBUG: market snapshot visibility ----
        logger.warning(
            "DEBUG snapshot | symbol=%s last_price=%s bars_count=%s day_volume=%s vwap=%s extra_keys=%s",
            sym,
            snapshot.last_price,
            len(snapshot.recent_bars or []),
            snapshot.day_volume,
            snapshot.vwap,
            list(snapshot.extra.keys()) if snapshot.extra else None,
        )

        if not snapshot.recent_bars:
            logger.error(
                "DEBUG NO BARS | symbol=%s snapshot_ts=%s extra=%s",
                sym,
                snapshot.ts,
                snapshot.extra,
            )
        else:
            b0 = snapshot.recent_bars[-1]
            logger.warning(
                "DEBUG LAST BAR | symbol=%s ts=%s O=%.2f H=%.2f L=%.2f C=%.2f V=%.2f",
                sym,
                b0.ts,
                b0.open,
                b0.high,
                b0.low,
                b0.close,
                b0.volume,
            )

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

        # 3) trade history (DB-backed)  ✅ FIX: needs symbol and is async
        hist_t0 = time.perf_counter()
        trade_history_records: List[Dict[str, Any]] = await self.trade_history_db.list_records(
            symbol=sym,
            limit=self.trade_history_limit,
        )
        hist_ms = (time.perf_counter() - hist_t0) * 1000.0

        # 4) build prompt
        prompt = build_user_prompt(
            snapshot,
            now_ts=now_ts or (snapshot.ts.strftime("%Y-%m-%d %H:%M:%S") if snapshot.ts else None),
            timezone_name=self.timezone_name,
            account=account,
            constraints=constraints,
            trade_history_records=trade_history_records,
        )

        # 5) call LLM
        llm_t0 = time.perf_counter()
        signal = await self.llm.chat_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            schema_hint=SIGNAL_SCHEMA_HINT,
        )
        llm_ms = (time.perf_counter() - llm_t0) * 1000.0

        # hard enforce horizon
        signal.horizon_minutes = 30

        # sizing normalization (cap / default)
        if self.fixed_amount_cny > 0:
            if signal.action in ("BUY", "SELL"):
                if float(signal.suggested_notional_cny) <= 0:
                    signal.suggested_notional_cny = self.fixed_amount_cny
                else:
                    signal.suggested_notional_cny = min(float(signal.suggested_notional_cny), self.fixed_amount_cny)
            else:
                signal.suggested_notional_cny = max(0.0, float(signal.suggested_notional_cny))

        total_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "signal_done | symbol=%s action=%s conf=%.3f dir=%s notional_cny=%.2f "
            "bars=%s hist=%s ms_total=%.1f ms_snap=%.1f ms_hist=%.1f ms_llm=%.1f last_price=%s",
            sym,
            signal.action,
            float(signal.confidence),
            signal.expected_direction,
            float(signal.suggested_notional_cny),
            len(snapshot.recent_bars or []),
            len(trade_history_records),
            total_ms,
            snap_ms,
            hist_ms,
            llm_ms,
            getattr(snapshot, "last_price", None),
        )

        return snapshot, signal

    async def get_signal(self, symbol: str, *, account: AccountState, now_ts: Optional[str] = None) -> TradeSignal:
        _, signal = await self._snapshot_and_signal(symbol, account=account, now_ts=now_ts)
        return signal

    async def maybe_execute(self, symbol: str, *, account: AccountState, now_ts: Optional[str] = None) -> TradeResult:
        t0 = time.perf_counter()
        sym = symbol.strip()

        snapshot, signal = await self._snapshot_and_signal(sym, account=account, now_ts=now_ts)

        # HOLD: no broker call
        if signal.action == "HOLD":
            logger.info("execute_skip_hold | symbol=%s conf=%.3f reason=%r", sym, float(signal.confidence), signal.reason)
            return TradeResult(
                ok=True,
                message="hold (no order placed)",
                ts=snapshot.ts or utcnow(),
                symbol=sym,
                action="HOLD",
                notional_cny=float(signal.suggested_notional_cny),
                paper=True,
                details={"signal": signal.model_dump(), "snapshot": snapshot.model_dump()},
            )

        ok, why = self.risk.can_trade(
            signal,
            account=account,
            snapshot=snapshot,
            suggested_notional_cny=float(signal.suggested_notional_cny),
        )

        if not ok:
            logger.warning(
                "execute_blocked | symbol=%s action=%s conf=%.3f why=%s",
                sym,
                signal.action,
                float(signal.confidence),
                why,
            )
            return TradeResult(
                ok=False,
                message=f"blocked: {why}",
                ts=snapshot.ts or utcnow(),
                symbol=sym,
                action=signal.action,
                notional_cny=float(signal.suggested_notional_cny),
                paper=True,
                details={"signal": signal.model_dump(), "snapshot": snapshot.model_dump()},
            )

        # broker order
        req = TradeRequest(symbol=sym, action=signal.action, notional_cny=float(signal.suggested_notional_cny))
        broker_t0 = time.perf_counter()
        res = await self.broker.place_order(req)
        broker_ms = (time.perf_counter() - broker_t0) * 1000.0

        res.details.setdefault("signal", signal.model_dump())
        res.details.setdefault("snapshot", snapshot.model_dump())

        logger.info(
            "execute_done | symbol=%s ok=%s action=%s notional_cny=%.2f broker_ms=%.1f msg=%r",
            sym,
            bool(res.ok),
            res.action,
            float(res.notional_cny),
            broker_ms,
            res.message,
        )

        if res.ok:
            self.risk.record_trade()

            # best-effort: persist feedback
            record = {
                "decision_ts": (snapshot.ts or utcnow()).strftime("%Y-%m-%d %H:%M:%S"),
                "action": signal.action,
                "expected_direction": signal.expected_direction,
                "confidence": float(signal.confidence),
                "executed_price_cny": None,
                "shares": None,
                "fees_cny": None,
                "realized_pnl_cny": None,
                "comment": f"Executed via broker | msg={res.message}",
            }
            try:
                await self.trade_history_db.append_record(symbol=sym, record=record)
            except Exception as e:
                logger.warning("trade_history_append_failed | symbol=%s err=%s: %s", sym, type(e).__name__, e)

        total_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("execute_total | symbol=%s ms_total=%.1f", sym, total_ms)

        return res
