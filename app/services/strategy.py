# app/services/strategy.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Tuple

from app.models import TradeSignal, TradeRequest, TradeResult, MarketSnapshot
from app.services.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from app.services.risk import RiskManager
from app.services.broker import Broker
from app.services.market_data import MarketDataProvider
from app.services.llm_client import OpenAICompatLLM

SIGNAL_SCHEMA_HINT = {
    "type": "object",
    "required": ["action", "horizon_minutes", "confidence", "reason", "suggested_notional_usd", "expected_direction"],
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
        fixed_amount_usd: float,
    ):
        self.market = market
        self.llm = llm
        self.risk = risk
        self.broker = broker
        self.fixed_amount_usd = float(fixed_amount_usd)

    async def _snapshot_and_signal(self, symbol: str) -> Tuple[MarketSnapshot, TradeSignal]:
        """
        Fetch one snapshot, build prompt once, ask LLM once.
        This avoids inconsistent timestamps between /signal and /execute.
        """
        snapshot = await self.market.get_snapshot(symbol)
        user_prompt = build_user_prompt(
            snapshot,
            fixed_amount=self.fixed_amount_usd,
            max_trades_left=self.risk.trades_left(),
        )

        signal = await self.llm.chat_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_hint=SIGNAL_SCHEMA_HINT,
        )

        # Enforce fixed notional
        signal.suggested_notional_usd = self.fixed_amount_usd
        # Ensure horizon is what you expect
        signal.horizon_minutes = 30

        return snapshot, signal

    async def get_signal(self, symbol: str) -> TradeSignal:
        _, signal = await self._snapshot_and_signal(symbol)
        return signal

    async def maybe_execute(self, symbol: str) -> TradeResult:
        snapshot, signal = await self._snapshot_and_signal(symbol)

        ok, why = self.risk.can_trade(signal)
        if not ok:
            return TradeResult(
                ok=False,
                message=f"blocked: {why}",
                ts=snapshot.ts or utcnow(),
                symbol=symbol,
                action=signal.action,
                notional_usd=signal.suggested_notional_usd,
                paper=True,
                details={
                    "signal": signal.model_dump(),
                    "snapshot": snapshot.model_dump(),
                },
            )

        req = TradeRequest(
            symbol=symbol,
            action=signal.action,
            notional_usd=signal.suggested_notional_usd,
        )

        res = await self.broker.place_order(req)
        res.details.setdefault("signal", signal.model_dump())
        res.details.setdefault("snapshot", snapshot.model_dump())

        if res.ok:
            self.risk.record_trade()

        return res
