# app/services/strategy.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Tuple, Optional

from app.models import TradeSignal, TradeRequest, TradeResult, MarketSnapshot, AccountState, ExecutionConstraints
from app.services.prompt_builder import SYSTEM_PROMPT, build_user_prompt 
from app.services.risk import RiskManager
from app.services.broker import Broker
from app.services.market_data import MarketDataProvider
from app.services.llm_client import OpenAICompatLLM
from app.services.trade_history import TradeHistoryStore, utcnow_str


SIGNAL_SCHEMA_HINT = {
    "type": "object",
    "required": [
        "action",
        "horizon_minutes",
        "confidence",
        "reason",
        "suggested_notional_usd",
        "expected_direction",
        "risk_notes",
    ],
    "properties": {
        "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "horizon_minutes": {"type": "integer", "enum": [30]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
        "suggested_notional_usd": {"type": "number", "minimum": 0.0},
        "expected_direction": {"type": "string", "enum": ["UP", "DOWN", "FLAT"]},
        "risk_notes": {"type": "string"},
    },
    "additionalProperties": False,
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StrategyEngine:
    def __init__(
        self,
        market: MarketDataProvider,
        llm: OpenAICompatLLM,
        risk: RiskManager,
        broker: Broker,
        *,
        fixed_amount_usd: float,
        trade_history: Optional[TradeHistoryStore] = None,
        timezone_name: str = "Asia/Shanghai",
        lot_size: int = 100,
        max_order_shares: Optional[int] = 1000,
        fees_bps_est: int = 5,
        slippage_bps_est: int = 5,
    ):
        self.market = market
        self.llm = llm
        self.risk = risk
        self.broker = broker

        # treat as a "cap" or "default" sizing hint; you can disable by setting <=0
        self.fixed_amount_usd = float(fixed_amount_usd)

        self.trade_history = trade_history or TradeHistoryStore()
        self.timezone_name = timezone_name
        self.lot_size = int(lot_size)
        self.max_order_shares = max_order_shares
        self.fees_bps_est = int(fees_bps_est)
        self.slippage_bps_est = int(slippage_bps_est)

    async def _snapshot_and_signal(
        self,
        symbol: str,
        *,
        account: AccountState,
        now_ts: Optional[str] = None,
    ) -> Tuple[MarketSnapshot, TradeSignal]:
        """
        Fetch one snapshot, build prompt once, ask LLM once.
        Avoid inconsistent timestamps between /signal and /execute.
        """
        snapshot = await self.market.get_snapshot(symbol)

        constraints = ExecutionConstraints(
            horizon_minutes=30,
            max_trades_left_today=self.risk.trades_left(),
            lot_size=self.lot_size,
            max_order_shares=self.max_order_shares,
            min_confidence_to_trade=self.risk.min_confidence,  # keep single source of truth
            fees_bps_est=self.fees_bps_est,
            slippage_bps_est=self.slippage_bps_est,
        )

        user_prompt = build_user_prompt(
            snapshot,
            now_ts=now_ts or (snapshot.ts.strftime("%Y-%m-%d %H:%M:%S") if snapshot.ts else None),
            timezone_name=self.timezone_name,
            account=account,
            constraints=constraints,
            trade_history_records=self.trade_history.list_records(),
        )

        signal = await self.llm.chat_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_hint=SIGNAL_SCHEMA_HINT,
        )

        # Hard enforce horizon
        signal.horizon_minutes = 30

        # Optional: cap or default notional (if you want to keep consistent sizing)
        if self.fixed_amount_usd > 0:
            # If model outputs something huge, cap it; if it outputs 0 for BUY/SELL, set default.
            if signal.action in ("BUY", "SELL"):
                if signal.suggested_notional_usd <= 0:
                    signal.suggested_notional_usd = self.fixed_amount_usd
                else:
                    signal.suggested_notional_usd = min(signal.suggested_notional_usd, self.fixed_amount_usd)
            else:
                # HOLD may be 0
                signal.suggested_notional_usd = max(0.0, float(signal.suggested_notional_usd))

        return snapshot, signal

    async def get_signal(self, symbol: str, *, account: AccountState, now_ts: Optional[str] = None) -> TradeSignal:
        _, signal = await self._snapshot_and_signal(symbol, account=account, now_ts=now_ts)
        return signal

    async def maybe_execute(self, symbol: str, *, account: AccountState, now_ts: Optional[str] = None) -> TradeResult:
        snapshot, signal = await self._snapshot_and_signal(symbol, account=account, now_ts=now_ts)

        # HOLD should be treated as a valid outcome (no broker call)
        if signal.action == "HOLD":
            return TradeResult(
                ok=True,
                message="hold (no order placed)",
                ts=snapshot.ts or utcnow(),
                symbol=symbol,
                action="HOLD",
                notional_usd=float(signal.suggested_notional_usd),
                paper=True,
                details={
                    "signal": signal.model_dump(),
                    "snapshot": snapshot.model_dump(),
                },
            )

        ok, why = self.risk.can_trade(signal)
        if not ok:
            return TradeResult(
                ok=False,
                message=f"blocked: {why}",
                ts=snapshot.ts or utcnow(),
                symbol=symbol,
                action=signal.action,
                notional_usd=float(signal.suggested_notional_usd),
                paper=True,
                details={
                    "signal": signal.model_dump(),
                    "snapshot": snapshot.model_dump(),
                },
            )

        # Place broker request (still notional-based)
        req = TradeRequest(
            symbol=symbol,
            action=signal.action,
            notional_usd=float(signal.suggested_notional_usd),
        )

        res = await self.broker.place_order(req)

        res.details.setdefault("signal", signal.model_dump())
        res.details.setdefault("snapshot", snapshot.model_dump())

        if res.ok:
            self.risk.record_trade()

            # Record feedback for next rounds (best-effort; if your broker later returns fill details, map them in)
            self.trade_history.append(
                {
                    "decision_ts": (snapshot.ts or utcnow()).strftime("%Y-%m-%d %H:%M:%S"),
                    "action": signal.action,
                    "expected_direction": signal.expected_direction,
                    "confidence": float(signal.confidence),
                    "executed_price_cny": None,      # fill later when you have it
                    "shares": None,                  # fill later when you size/execute shares
                    "fees_cny": None,                # fill later
                    "realized_pnl_cny": None,         # fill later
                    "comment": f"Executed via broker | msg={res.message}",
                }
            )

        return res
