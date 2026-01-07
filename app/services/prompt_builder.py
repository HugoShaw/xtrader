# app/services/prompt_builder.py
from __future__ import annotations

import json

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from app.logging_config import logger
from app.models import MarketSnapshot, AccountState, ExecutionConstraints

SYSTEM_PROMPT = """You are a cautious quantitative trading assistant for China A-share intraday trading.

You MUST output ONLY valid JSON.
- No markdown
- No code fences
- No explanations outside JSON
- No extra keys beyond the schema below

You MUST use ONLY the information provided by the user:
- 5-minute OHLCV price bars
- Account state
- Execution constraints
- Trade history (your past decisions + realized PnL/cost)
Do NOT assume any news, fundamentals, or external data.

=== Trading cadence ===
- This decision is made once every 30 minutes during market trading hours.
- You are given 5-minute OHLCV bars up to the latest completed bar before "now".
- You must make a forward-looking decision for the NEXT 30 minutes.

=== How to use trade history ===
- Trade history is feedback about how previous decisions performed.
- If recent trades show consistent losses under similar patterns, reduce aggressiveness:
  - Prefer HOLD more often
  - Lower confidence
  - Use smaller suggested_notional_cny
- If trade history is missing, short, or inconsistent, do NOT overfit; stay conservative.

=== Output schema (STRICT) ===
Return a JSON object with EXACTLY these keys:
- action: one of ["BUY", "SELL", "HOLD"]
- horizon_minutes: must be 30
- confidence: number between 0 and 1
- expected_direction: one of ["UP", "DOWN", "FLAT"]
- suggested_notional_cny: number >= 0 (required)
- reason: short, specific, data-grounded explanation referencing recent bars/volume/volatility and (if helpful) recent trade outcomes
- risk_notes: short, include at least ONE concrete risk

You MUST NOT output any other fields.

=== Execution rules ===
1) You MUST respect all constraints:
   - max_trades_left_today
   - cannot BUY if available cash is insufficient
   - cannot SELL more shares than current position

2) BUY means: buy using suggested_notional_cny as a sizing hint.
3) SELL means: sell using suggested_notional_cny as a sizing hint.
4) HOLD means: suggested_notional_cny can be 0.
5) If max_trades_left_today <= 0 → MUST return HOLD.
6) If confidence < min_confidence_to_trade → MUST return HOLD.

=== Risk & conservatism ===
- Be conservative.
- If signals conflict, volatility is elevated, liquidity degrades, or trade_history indicates recent poor performance → HOLD or smaller size.
- Prefer smaller position changes over aggressive flipping.
- You are NOT guaranteed to be correct.

Return ONLY the JSON object.
"""

_SH_TZ = ZoneInfo("Asia/Shanghai")

def _fmt_ts_shanghai(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_SH_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _bars_to_5m_ohlcv(snapshot: MarketSnapshot) -> List[Dict[str, Any]]:
    bars = []
    for b in (snapshot.recent_bars or []):
        item: Dict[str, Any] = {
            "ts": _fmt_ts_shanghai(b.ts),
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": float(b.volume),
        }

        amt = getattr(b, "amount", None)
        if amt is not None:
            try:
                item["amount"] = float(amt)
            except Exception:
                pass

        vwap = getattr(b, "vwap", None)
        if vwap is not None:
            try:
                item["vwap"] = float(vwap)
            except Exception:
                pass

        bars.append(item)

    bars.sort(key=lambda x: x["ts"])
    return bars


def build_user_prompt(
    snapshot: MarketSnapshot,
    *,
    now_ts: Optional[str],
    timezone_name: str,
    account: AccountState,
    constraints: ExecutionConstraints,
    trade_history_records: List[Dict[str, Any]],
    # Policy knobs (optional, but useful to keep stable)
    decision_goal: str = "Prefer low volatility and controlled trading frequency",
    trade_style: str = "Intraday trading, decision every 30 minutes",
) -> str:
    """
    Returns JSON string used as req.user for /api/llm/chat (in JSON schema mode).
    """

    payload: Dict[str, Any] = {
        "task": "Make a position- and cash-aware trade decision for the next 30 minutes using 5-minute OHLCV bars + trade history feedback",
        "now": {
            "ts": now_ts or (_fmt_ts_shanghai(snapshot.ts) if snapshot.ts else ""),
            "timezone": timezone_name,
            "note": "Decision time is aligned to the latest completed 5-minute bar <= now.",
        },
        "instrument": {
            "symbol": snapshot.symbol,
            "market": "CN-A",
            "bar_freq_minutes": 5,
            "currency": "CNY",
        },
        "account_state": {
            "cash_cny": float(account.cash_cny),
            "position_shares": int(account.position_shares),
            "avg_cost_cny": float(account.avg_cost_cny) if account.avg_cost_cny is not None else None,
            "unrealized_pnl_cny": float(account.unrealized_pnl_cny) if account.unrealized_pnl_cny is not None else None,
        },
        "execution_constraints": {
            "horizon_minutes": int(constraints.horizon_minutes),
            "max_trades_left_today": int(constraints.max_trades_left_today),
            "lot_size": int(constraints.lot_size),
            "max_order_shares": int(constraints.max_order_shares) if constraints.max_order_shares is not None else None,
            "min_confidence_to_trade": float(constraints.min_confidence_to_trade),
            "fees_bps_est": int(constraints.fees_bps_est),
            "slippage_bps_est": int(constraints.slippage_bps_est),
        },
        "decision_policy": {
            "goal": decision_goal,
            "trade_style": trade_style,
            # "rule": "If you expect price to rise in the next 30 minutes → SELL (size via suggested_notional_cny). If you expect price to fall → BUY (size via suggested_notional_cny). Otherwise HOLD.",
            "rule": "If you expect price to rise in the next 30 minutes → BUY (size via suggested_notional_cny). If you expect price to fall → SELL (size via suggested_notional_cny). Otherwise HOLD.",
            "position_management": "Scale in/out conservatively; avoid aggressive flips.",
        },
        "trade_history": {
            "note": "All prior trades were executed based on your previous outputs. Use this as feedback to self-correct sizing/confidence (do not add new output keys).",
            "records": trade_history_records,
        },
        "market_data": {
            "bars_5m": _bars_to_5m_ohlcv(snapshot),
        },
        "output_required": {
            "return_only_json": True,
            "horizon_minutes": 30,
        },
        "important_rules": [
            "Use the LAST close price as current reference price",
            "If confidence < min_confidence_to_trade OR max_trades_left_today <= 0 -> HOLD",
            "Use trade_history as feedback to adjust confidence and suggested_notional_cny conservatively",
        ],
    }

    # Drop None fields for compactness
    def _drop_none(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _drop_none(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, list):
            return [_drop_none(v) for v in obj]
        return obj

    payload = _drop_none(payload)

    logger.warning(
        "PROMPT DATA | bars=%s trade_history=%s account_cash=%.2f",
        len(snapshot.recent_bars or []),
        len(trade_history_records or []),
        float(account.cash_cny),
    )

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
