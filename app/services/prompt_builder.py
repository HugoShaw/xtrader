# app/services/prompt_builder.py
from __future__ import annotations

import json
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from app.config import settings
from app.logging_config import logger
from app.models import MarketSnapshot, AccountState, ExecutionConstraints
  
_SH_TZ = ZoneInfo("Asia/Shanghai")

# =====================================================================
# Contract change:
# - Size is in LOTS (手), not money
# - Only intraday trade history (today) is provided
# =====================================================================
SYSTEM_PROMPT = f"""You are a quantitative trading signal assistant for China A-share intraday trading.

You MUST output ONLY valid JSON.
- No markdown
- No code fences
- No explanations outside JSON
- No extra keys beyond the schema below

You MUST use ONLY the information provided by the user:
- 5-minute OHLCV price bars
- Derived features computed from those bars
- Account state
- Execution constraints (including lot_size)
- Transaction costs (buy_fee_rate, sell_fee_rate) and slippage estimate
- Intraday trade history (TODAY only: your past decisions + realized PnL/cost)
Do NOT assume any news, fundamentals, or external data.

=== Trading cadence ===
- This decision is made once every 30 minutes during market trading hours.
- You are given 5-minute OHLCV bars up to the latest completed bar before "now".
- You must produce a forward-looking signal for the NEXT 30 minutes.

=== Transaction costs (MUST consider) ===
- Buying total fee rate = {settings.buy_fee_rate:.5f} (0.015% of notional)
- Selling total fee rate = {settings.sell_fee_rate:.4f} (0.25% of notional)
- Slippage is provided as slippage_bps_est in execution_constraints.
- If expected move is too small to overcome costs + slippage, reduce confidence and/or suggested_lots.

=== IMPORTANT: Signal vs Execution ===
- You output a *signal* (BUY/SELL/HOLD) and a LOTS sizing hint.
- The system will apply final execution gating (cash/position/trade limits/confidence thresholds).
- Therefore: Do NOT force HOLD just because of constraints; still output the best signal you see,
  but keep suggested_lots conservative when uncertainty is high.

=== How to use intraday trade history (TODAY only) ===
- Intraday history is feedback about how previous decisions performed today (same-day microstructure).
- If recent trades show consistent losses under similar patterns, reduce aggressiveness:
  - Lower confidence
  - Use smaller suggested_lots
  - HOLD when edge is unclear or fee-negative
- If intraday history is missing/short, do NOT overfit.

=== Range (FLAT) regime is tradable ===
- expected_direction describes the most likely *drift* over next 30 minutes.
- If expected_direction is FLAT, you may still output BUY or SELL as mean-reversion:
  - If price is near recent range low (pos_in_range_0_1 small) and no breakdown signs -> BUY (small lots)
  - If price is near recent range high (pos_in_range_0_1 large) and no breakout signs -> SELL (small lots)
  - Otherwise -> HOLD

=== Output schema (STRICT) ===
Return a JSON object with EXACTLY these keys:
- action: one of ["BUY", "SELL", "HOLD"]
- horizon_minutes: must be 30
- confidence: number between 0 and 1 (calibrated; higher only when edge is clear and fee-aware)
- expected_direction: one of ["UP", "DOWN", "FLAT"]
- suggested_lots: integer >= 0 (required; number of lots/手; MUST be multiple-lot thinking; 0 for HOLD)
- reason: short, specific, data-grounded explanation referencing derived_features and recent bars and (if helpful) intraday trade history; mention cost-awareness when relevant
- risk_notes: short, include at least ONE concrete risk

You MUST NOT output any other fields.

Return ONLY the JSON object.
"""


def _fmt_ts_shanghai(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_SH_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _parse_shanghai_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.strptime(str(s).strip(), "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=_SH_TZ)
    except Exception:
        return None


def _bars_to_5m_ohlcv(snapshot: MarketSnapshot) -> List[Dict[str, Any]]:
    bars: List[Dict[str, Any]] = []
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


def _compute_micro_features(bars: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(bars) < 6:
        return {"note": "insufficient_bars_for_features", "n_bars": len(bars)}

    closes = [float(b["close"]) for b in bars]
    highs = [float(b["high"]) for b in bars]
    lows = [float(b["low"]) for b in bars]
    vols = [float(b.get("volume", 0.0)) for b in bars]

    def pct(a: float, b: float) -> float:
        return (a / b - 1.0) if b else 0.0

    last = closes[-1]
    c_1 = closes[-2]
    c_3 = closes[-4]
    c_6 = closes[-7] if len(closes) >= 7 else closes[0]

    window = 12 if len(closes) >= 13 else 6
    start = max(1, len(closes) - window)
    rets = [pct(closes[i], closes[i - 1]) for i in range(start, len(closes))]
    mean_ret = sum(rets) / max(len(rets), 1)
    rv = (sum((r - mean_ret) ** 2 for r in rets) / max(len(rets), 1)) ** 0.5

    N = min(12, len(closes))
    hiN = max(highs[-N:])
    loN = min(lows[-N:])
    rangeN = max(hiN - loN, 1e-9)
    pos_in_range = (last - loN) / rangeN
    range_pct = rangeN / max(last, 1e-9)

    avg_vol = (sum(vols[-N:]) / max(N, 1)) if N else 0.0
    recent_vol = sum(vols[-3:]) / 3.0
    vol_ratio = (recent_vol / avg_vol) if avg_vol > 0 else 1.0

    return {
        "n_bars": len(bars),
        "last_close": float(last),
        "ret_5m": float(pct(last, c_1)),
        "ret_15m": float(pct(last, c_3)),
        "ret_30m": float(pct(last, c_6)) if c_6 else None,
        "realized_vol": float(rv),
        "range_high_N": float(hiN),
        "range_low_N": float(loN),
        "range_pct_N": float(range_pct),
        "pos_in_range_0_1": float(pos_in_range),
        "volume_ratio_recent": float(vol_ratio),
    }


def _intraday_today_only(
    records: List[Dict[str, Any]],
    *,
    now_ts: Optional[str],
) -> Tuple[date, List[Dict[str, Any]]]:
    """
    Returns:
      - trading_day (Shanghai date)
      - intraday_today_records (sorted asc by decision_ts)
    """
    now_dt = _parse_shanghai_ts(now_ts) or datetime.now(_SH_TZ)
    trading_day = now_dt.date()

    tmp: List[Tuple[datetime, Dict[str, Any]]] = []
    for r in records or []:
        dt = _parse_shanghai_ts(r.get("decision_ts"))
        if dt is None:
            continue
        if dt.date() != trading_day:
            continue
        tmp.append((dt, r))

    tmp.sort(key=lambda x: x[0])  # asc
    today = [r for _, r in tmp]
    return trading_day, today


def _summarize_intraday(records_today: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records_today:
        return {"note": "no_intraday_records_today"}

    pnls: List[float] = []
    fees: List[float] = []
    actions: List[str] = []

    win = 0
    loss = 0
    flat = 0

    for r in records_today:
        a = str(r.get("action") or "").upper().strip()
        if a:
            actions.append(a)

        pnl = r.get("realized_pnl_cny")
        fee = r.get("fees_cny")

        if pnl is not None:
            try:
                p = float(pnl)
                pnls.append(p)
                if p > 0:
                    win += 1
                elif p < 0:
                    loss += 1
                else:
                    flat += 1
            except Exception:
                pass

        if fee is not None:
            try:
                fees.append(float(fee))
            except Exception:
                pass

    total_pnl = sum(pnls) if pnls else None
    total_fees = sum(fees) if fees else None
    last_action = actions[-1] if actions else None

    return {
        "n_trades_today": len(records_today),
        "wins_today": win,
        "losses_today": loss,
        "flat_today": flat,
        "total_realized_pnl_cny_today": total_pnl,
        "total_fees_cny_today": total_fees,
        "last_action_today": last_action,
        "note": "Use these TODAY records as highest-priority feedback for confidence + suggested_lots. Avoid repeating losing patterns.",
    }


def build_user_prompt(
    snapshot: MarketSnapshot,
    *,
    now_ts: Optional[str],
    timezone_name: str,
    account: AccountState,
    constraints: ExecutionConstraints,
    trade_history_records: List[Dict[str, Any]],
    decision_goal: str = "Fee-aware intraday trading with controlled trading frequency",
    trade_style: str = "Intraday trading, decision every 30 minutes",
) -> str:
    bars = _bars_to_5m_ohlcv(snapshot)
    features = _compute_micro_features(bars)
    last_close = features.get("last_close")

    trading_day, intraday_today_records = _intraday_today_only(
        trade_history_records or [],
        now_ts=now_ts,
    )
    intraday_summary = _summarize_intraday(intraday_today_records)

    INTRADAY_MAX = settings.max_trades_per_day

    payload: Dict[str, Any] = {
        "task": "Produce a 30-minute trading SIGNAL (BUY/SELL/HOLD) using 5-min OHLCV + derived_features + fee model; size MUST be in lots (手). Final execution gating is handled by the system.",
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
            "current_reference_price_close": float(last_close) if last_close is not None else None,
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
            # LOT sizing contract: suggested_lots will be multiplied by this lot_size to get shares
            "lot_size": int(constraints.lot_size),
            "max_order_shares": int(constraints.max_order_shares) if constraints.max_order_shares is not None else None,
            "min_confidence_to_trade": float(constraints.min_confidence_to_trade),
            "slippage_bps_est": int(constraints.slippage_bps_est),
            "sizing_rule": "Output suggested_lots (手). Shares = suggested_lots * lot_size. For HOLD use suggested_lots=0.",
        },
        "transaction_costs": {
            "buy_fee_rate": settings.buy_fee_rate,
            "sell_fee_rate": settings.sell_fee_rate,
            "note": "Fees apply to notional. Edge must overcome fees + slippage for confidence to be high.",
        },
        "decision_policy": {
            "goal": decision_goal,
            "trade_style": trade_style,
            "signal_rule": "UP drift -> BUY; DOWN drift -> SELL; FLAT drift may still BUY/SELL as mean-reversion if price is near range extremes; otherwise HOLD.",
            "position_management": "Scale in/out conservatively; avoid aggressive flips.",
            "cost_awareness": "If the expected move is likely fee-negative after costs+slippage, lower confidence and/or suggested_lots or HOLD.",
            "note": "Final execution gating (confidence threshold, trade limits, cash/position feasibility) is done outside the LLM.",
        }, 
        "intraday_trade_history": {
            "trading_day_shanghai": trading_day.isoformat(),
            "summary": intraday_summary,
            "records_today": intraday_today_records[-INTRADAY_MAX:],
            "how_to_use": [
                "Use this TODAY feedback to adjust confidence and suggested_lots.",
                "If today is losing, reduce suggested_lots aggressively or HOLD unless edge is clear and fee-aware.",
            ],
        }, 
        "market_data": {"bars_5m": bars},
        "derived_features": features,
        "output_required": {
            "return_only_json": True,
            "horizon_minutes": 30,
            "size_unit": "lots",
        },
        "important_rules": [
            "Size MUST be suggested_lots (integer lots/手). Do not output money sizing.",
            "Keep confidence calibrated and fee-aware",
            "Range regime is tradable via mean-reversion near range extremes (use pos_in_range_0_1)",
            "Do not hallucinate: use only provided bars + derived_features + intraday_history",
        ],
    }

    def _drop_none(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _drop_none(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, list):
            return [_drop_none(v) for v in obj]
        return obj

    payload = _drop_none(payload)

    logger.warning(
        "PROMPT DATA | bars=%s intraday_today=%s cash=%.2f lot_size=%s",
        len(snapshot.recent_bars or []),
        len(intraday_today_records or []),
        float(account.cash_cny),
        int(constraints.lot_size),
    )

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
