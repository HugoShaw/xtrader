# app/services/prompt_builder.py
from __future__ import annotations

import json
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple, Literal

from app.config import settings
from app.logging_config import logger
from app.models import MarketSnapshot, AccountState, ExecutionConstraints
from app.utils.timeutils import (
    now_shanghai,
    ensure_shanghai,
    fmt_shanghai,
    parse_shanghai,
)

StrategyMode = Literal["mild", "normal", "aggressive"]


# =====================================================================
# Contract:
# - Bars are intraday OHLCV with dynamic bar_freq_minutes
# - Size is in LOTS (手), not money
# - Only intraday trade history (today) is provided
# - LONG-only semantics (SELL reduces existing long; no shorting)
# =====================================================================

SYSTEM_PROMPT = f"""You are a quantitative trading SIGNAL assistant for China A-share intraday trading.

You MUST output ONLY valid JSON:
- No markdown
- No code fences
- No explanations outside JSON
- No extra keys beyond the schema below

You MUST use ONLY the information provided by the user:
- Intraday OHLCV bars (frequency is instrument.bar_freq_minutes)
- Derived features computed from those bars (derived_features)
- Account state
- Execution constraints (including lot_size, max_order_shares, max_trades_left_today, min_confidence_to_trade)
- Transaction costs (buy_fee_rate, sell_fee_rate) and bps estimates (fees_bps_est, slippage_bps_est)
- Intraday trade history (TODAY only)

Do NOT assume any news, fundamentals, or external data.

=== Trading cadence ===
- This decision is made once every 30 minutes during market trading hours.
- You are given bars up to the latest completed bar before "now".
- You must produce a forward-looking SIGNAL for the NEXT 30 minutes.

=== IMPORTANT: what features mean (avoid common mistake) ===
- derived_features.ret_15m / ret_30m / ret_horizon are BACKWARD-LOOKING realized returns (momentum proxies).
  They are NOT guaranteed forecasts.
- Form an expected 30-minute drift estimate by combining:
  (A) momentum proxies (ret_15m/ret_30m/ret_horizon),
  (B) range position (pos_in_range_0_1),
  (C) volatility/regime (realized_vol_recent, range_pct_N),
  (D) volume confirmation (volume_ratio_recent),
  (E) intraday feedback (today trade history).

=== Transaction costs (MUST consider carefully) ===
Given constants:
- buy_fee_rate (fraction of notional) = {settings.buy_fee_rate:.6f}
- sell_fee_rate (fraction of notional) = {settings.sell_fee_rate:.6f}
And bps estimates in execution_constraints:
- fees_bps_est, slippage_bps_est (1 bps = 0.0001)

Interpretation:
- One-way BUY friction ≈ buy_fee_rate + (fees_bps_est + slippage_bps_est)/1e4
- One-way SELL friction ≈ sell_fee_rate + (fees_bps_est + slippage_bps_est)/1e4
- Round-trip friction ≈ buy_one_way + sell_one_way

Rule:
- Use transaction_costs.round_trip_friction_rate_est as the conservative hurdle.
- Do NOT trade on a tiny edge that barely clears costs: require margin because signals are noisy.

=== Long-only semantics ===
- BUY means increasing/entering a long position.
- SELL means reducing/closing an EXISTING long position (no shorting).
- If position_shares == 0, prefer HOLD over SELL unless you have a strong bearish drift AND the system explicitly supports shorting (assume it does NOT).

=== Signal vs Execution (to avoid confusing behavior) ===
- You output: action + confidence + expected_direction + suggested_lots.
- The system applies final execution gating (cash/position feasibility, max trades left, confidence thresholds).
- If constraints likely BLOCK execution (e.g., max_trades_left_today == 0), you may still output the best direction,
  but set suggested_lots = 0 and mention "blocked by constraints" briefly in reason.

=== Confidence calibration (anchors) ===
- 0.80–0.95: strong edge with clear confirmation; fee-positive with margin.
- 0.65–0.80: moderate edge; costs likely covered; some confirmation.
- 0.50–0.65: marginal / mixed; borderline after costs; size tiny or HOLD.
- <0.50: unclear or likely fee-negative; HOLD (or 0 lots).

=== Anti-churn / consistency ===
- Avoid flip-flopping within the same day (BUY->SELL or SELL->BUY) unless evidence is strong (range break, momentum reversal, clear regime shift).
- If last 2–3 trades are negative today, reduce confidence and size unless current setup is clearly stronger.

=== Range (FLAT) regime is tradable ===
- expected_direction is the most likely drift (UP/DOWN/FLAT).
- If drift is FLAT, you may still BUY/SELL as mean-reversion:
  - pos_in_range_0_1 low + no breakdown signs -> BUY (mean reversion)
  - pos_in_range_0_1 high + no breakout signs -> SELL (reduce long)
  - else HOLD

=== Strategy mode (authoritative) ===
You MUST follow decision_policy.strategy_mode and its multipliers:
- Mild: stricter hurdle, stronger confirmation, smaller size, fewer flips.
- Normal: balanced.
- Aggressive: does NOT mean "trade on marginal edge"; it means "size up when confirmed".

=== Sizing guidance (fee-aware) ===
- suggested_lots must be integer >= 0 (0 for HOLD).
- Shares = suggested_lots * lot_size.
- Respect max_order_shares when present.
- Size is driven by net edge (after costs), regime/volatility, and TODAY feedback.
- If edge is borderline after costs -> HOLD or suggested_lots = 0–1 with low confidence.
- If edge clears costs with margin AND confirmed -> larger lots may be appropriate (within constraints).

=== Reason must be data-grounded ===
- reason must cite at least TWO of:
  derived_features.ret_horizon/ret_30m/ret_15m, pos_in_range_0_1, volume_ratio_recent, realized_vol_recent, range_pct_N,
  and/or intraday trade history summary (wins/losses/last_action).
- Mention fee-awareness when relevant (e.g., "edge barely clears round-trip friction" -> reduce size).

=== Output schema (STRICT) ===
Return a JSON object with EXACTLY these keys:
- action: one of ["BUY", "SELL", "HOLD"]
- horizon_minutes: must be 30
- confidence: number between 0 and 1
- expected_direction: one of ["UP", "DOWN", "FLAT"]
- suggested_lots: integer >= 0 (0 for HOLD)
- reason: short, specific, data-grounded explanation
- risk_notes: short, include at least ONE concrete risk

You MUST NOT output any other fields.
Return ONLY the JSON object.
"""


def _fmt_ts(dt: Optional[datetime]) -> str:
    """Always format in Shanghai, 'YYYY-MM-DD HH:MM:SS'."""
    s = fmt_shanghai(ensure_shanghai(dt) if dt else None)
    return s or ""


def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    """Parse Shanghai 'YYYY-MM-DD HH:MM:SS' -> tz-aware datetime."""
    return parse_shanghai(s)


def _as_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            v = int(x)
            return v if v > 0 else None
        s = str(x).strip()
        if not s:
            return None
        v = int(float(s))
        return v if v > 0 else None
    except Exception:
        return None


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        v = float(x)
        return v if (v == v) else None  # NaN guard
    except Exception:
        return None


def _infer_bar_freq_minutes(snapshot: MarketSnapshot, default: int = 5) -> int:
    """
    Best-effort inference of bar frequency (minutes) from snapshot.extra.

    Supported shapes:
      snapshot.extra["bars_window_cn"]["period"] = "5" / "1" / "15" ...
      snapshot.extra["bars_window_cn"]["period_min"] = 5
      snapshot.extra["bar_freq_minutes"] = 5
      snapshot.extra["min_period"] = "5"
    """
    extra = getattr(snapshot, "extra", None) or {}
    if not isinstance(extra, dict):
        return int(default)

    bw = extra.get("bars_window_cn")
    if isinstance(bw, dict):
        v = _as_int(bw.get("period")) or _as_int(bw.get("period_min"))
        if v:
            return int(v)

    v = (
        _as_int(extra.get("bar_freq_minutes"))
        or _as_int(extra.get("min_period"))
        or _as_int(extra.get("min_period_minutes"))
    )
    if v:
        return int(v)

    return int(default)


def _bars_to_ohlcv(snapshot: MarketSnapshot) -> List[Dict[str, Any]]:
    """
    Convert snapshot.recent_bars into JSON-safe OHLCV list.
    NOTE: bar frequency is NOT encoded here; caller should add instrument.bar_freq_minutes.
    """
    bars: List[Dict[str, Any]] = []
    for b in (snapshot.recent_bars or []):
        try:
            ts_s = _fmt_ts(getattr(b, "ts", None))
            o = _as_float(getattr(b, "open", None))
            h = _as_float(getattr(b, "high", None))
            l = _as_float(getattr(b, "low", None))
            c = _as_float(getattr(b, "close", None))
            v = _as_float(getattr(b, "volume", 0.0)) or 0.0
            if c is None:
                continue

            item: Dict[str, Any] = {
                "ts": ts_s,
                "open": float(o) if o is not None else float(c),
                "high": float(h) if h is not None else float(c),
                "low": float(l) if l is not None else float(c),
                "close": float(c),
                "volume": float(v),
            }

            amt = _as_float(getattr(b, "amount", None))
            if amt is not None:
                item["amount"] = float(amt)

            vwap = _as_float(getattr(b, "vwap", None))
            if vwap is not None:
                item["vwap"] = float(vwap)

            bars.append(item)
        except Exception:
            continue

    bars.sort(key=lambda x: x["ts"])  # lex sort works due to fixed ts format
    return bars


def _compute_micro_features(
    bars: List[Dict[str, Any]],
    *,
    bar_freq_minutes: int,
    horizon_minutes: int = 30,
) -> Dict[str, Any]:
    """
    Compute features that adapt to bar frequency.

    NOTE:
      - Returns are backward-looking realized returns (momentum proxies).
      - This function does not "forecast" the next 30 minutes.
    """
    bar_freq_minutes = max(int(bar_freq_minutes or 1), 1)
    horizon_minutes = max(int(horizon_minutes or 30), 1)

    if len(bars) < 6:
        return {
            "note": "insufficient_bars_for_features",
            "n_bars": len(bars),
            "bar_freq_minutes": bar_freq_minutes,
            "horizon_minutes": horizon_minutes,
        }

    closes = [float(b["close"]) for b in bars]
    highs = [float(b["high"]) for b in bars]
    lows = [float(b["low"]) for b in bars]
    vols = [float(b.get("volume", 0.0)) for b in bars]

    def pct(a: float, b: float) -> float:
        return (a / b - 1.0) if b else 0.0

    def idx_back(k: int) -> int:
        return max(0, len(closes) - 1 - max(int(k), 0))

    last = closes[-1]

    c_1 = closes[idx_back(1)]
    c_3 = closes[idx_back(3)]
    c_6 = closes[idx_back(6)]

    bars_15m = max(1, int(round(15 / bar_freq_minutes)))
    bars_30m = max(1, int(round(30 / bar_freq_minutes)))
    bars_hz = max(1, int(round(horizon_minutes / bar_freq_minutes)))

    c_15m = closes[idx_back(bars_15m)]
    c_30m = closes[idx_back(bars_30m)]
    c_hz = closes[idx_back(bars_hz)]

    # Realized vol over ~60m window (or min 6 bars)
    bars_60m = max(6, int(round(60 / bar_freq_minutes)))
    window = min(max(6, bars_60m), len(closes) - 1)
    start = max(1, len(closes) - window)
    rets = [pct(closes[i], closes[i - 1]) for i in range(start, len(closes))]
    mean_ret = sum(rets) / max(len(rets), 1)
    rv = (sum((r - mean_ret) ** 2 for r in rets) / max(len(rets), 1)) ** 0.5

    # Range window ~60m (min 12 bars)
    N_pref = max(12, int(round(60 / bar_freq_minutes)))
    N = min(N_pref, len(closes))
    hiN = max(highs[-N:])
    loN = min(lows[-N:])
    rangeN = max(hiN - loN, 1e-9)
    pos_in_range = (last - loN) / rangeN
    pos_in_range = max(0.0, min(1.0, float(pos_in_range)))
    range_pct = rangeN / max(last, 1e-9)

    # Volume ratio: recent vs average in the range window
    recent_n = min(max(3, bars_15m), len(vols))
    avg_vol = (sum(vols[-N:]) / max(N, 1)) if N else 0.0
    recent_vol = (sum(vols[-recent_n:]) / max(recent_n, 1)) if recent_n else 0.0
    vol_ratio = (recent_vol / avg_vol) if avg_vol > 0 else 1.0

    return {
        "n_bars": len(bars),
        "bar_freq_minutes": int(bar_freq_minutes),
        "horizon_minutes": int(horizon_minutes),
        "last_close": float(last),
        "ret_1bar": float(pct(last, c_1)),
        "ret_3bar": float(pct(last, c_3)),
        "ret_6bar": float(pct(last, c_6)),
        "ret_15m": float(pct(last, c_15m)),
        "ret_30m": float(pct(last, c_30m)),
        "ret_horizon": float(pct(last, c_hz)),
        "realized_vol_recent": float(rv),
        "range_high_N": float(hiN),
        "range_low_N": float(loN),
        "range_pct_N": float(range_pct),
        "pos_in_range_0_1": float(pos_in_range),
        "volume_ratio_recent": float(vol_ratio),
        "volume_recent_n_bars": int(recent_n),
        "range_window_n_bars": int(N),
        "note": "All ret_* fields are backward-looking realized returns (momentum proxies), not forecasts.",
    }


def _intraday_today_only(
    records: List[Dict[str, Any]],
    *,
    now_ts: Optional[str],
) -> Tuple[date, List[Dict[str, Any]]]:
    now_dt = _parse_ts(now_ts) or now_shanghai()
    now_dt = ensure_shanghai(now_dt)
    trading_day = now_dt.date()

    tmp: List[Tuple[datetime, Dict[str, Any]]] = []
    for r in records or []:
        dt = _parse_ts(r.get("decision_ts"))
        if dt is None:
            continue
        dt = ensure_shanghai(dt)
        if dt.date() != trading_day:
            continue
        tmp.append((dt, r))

    tmp.sort(key=lambda x: x[0])
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
        "note": (
            "Use TODAY records as highest-priority feedback for confidence and sizing. "
            "If losing today, reduce suggested_lots and confidence unless current setup is clearly stronger."
        ),
    }


def _cost_model_from_constraints(constraints: ExecutionConstraints) -> Dict[str, Any]:
    """
    Build a fee/slippage model the LLM can reason about numerically.
    Exposes friction rates (fraction of notional) and bps approximations.
    """
    fees_bps = int(getattr(constraints, "fees_bps_est", 0) or 0)
    slip_bps = int(getattr(constraints, "slippage_bps_est", 0) or 0)

    fees_bps = max(0, min(fees_bps, 50_000))
    slip_bps = max(0, min(slip_bps, 50_000))
    add_bps = fees_bps + slip_bps
    add_rate = add_bps / 1e4

    buy_fee_rate = float(settings.buy_fee_rate)
    sell_fee_rate = float(settings.sell_fee_rate)

    buy_oneway_rate = buy_fee_rate + add_rate
    sell_oneway_rate = sell_fee_rate + add_rate
    roundtrip_rate = buy_oneway_rate + sell_oneway_rate

    def to_bps(rate: float) -> float:
        return float(rate) * 1e4

    return {
        "buy_fee_rate": buy_fee_rate,
        "sell_fee_rate": sell_fee_rate,
        "fees_bps_est": int(fees_bps),
        "slippage_bps_est": int(slip_bps),
        "additive_bps_est_for_hurdle": int(add_bps),
        "buy_one_way_friction_rate_est": float(buy_oneway_rate),
        "sell_one_way_friction_rate_est": float(sell_oneway_rate),
        "round_trip_friction_rate_est": float(roundtrip_rate),
        "buy_one_way_friction_bps_est": float(to_bps(buy_oneway_rate)),
        "sell_one_way_friction_bps_est": float(to_bps(sell_oneway_rate)),
        "round_trip_friction_bps_est": float(to_bps(roundtrip_rate)),
        "note": "Use round_trip_friction_* as a conservative hurdle; require margin above costs.",
    }


def _normalize_strategy_mode(mode: Any) -> StrategyMode:
    s = str(mode or "normal").strip().lower()
    if s in ("mild", "normal", "aggressive"):
        return s  # type: ignore[return-value]
    return "normal"


def _drop_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _drop_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_none(v) for v in obj]
    return obj


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
    strategy_mode: StrategyMode = "normal",
) -> str:
    """
    Build JSON payload for the LLM.

    Notes:
      - market_contract is carried inside constraints.market_contract (dict)
      - exposes explicit cost model to force fee-aware reasoning
      - makes strategy_mode behavior more robust: aggressive scales size when confirmed,
        rather than lowering the cost hurdle into noise-churn territory.
    """
    snap_ts = ensure_shanghai(snapshot.ts) if snapshot.ts else None
    strategy_mode = _normalize_strategy_mode(strategy_mode)

    # market_contract from constraints (backward compatible)
    mc: Dict[str, Any] = {}
    try:
        v = getattr(constraints, "market_contract", None)
        if isinstance(v, dict):
            mc = dict(v)
    except Exception:
        mc = {}

    min_period = mc.get("min_period")
    lookback_minutes = mc.get("min_lookback_minutes")

    # Prefer explicit contract, fallback to snapshot.extra inference, fallback default
    bar_freq = (
        int(float(min_period))
        if (min_period is not None and str(min_period).strip().replace(".", "", 1).isdigit())
        else None
    ) or _infer_bar_freq_minutes(snapshot, default=5)

    bars = _bars_to_ohlcv(snapshot)
    features = _compute_micro_features(
        bars,
        bar_freq_minutes=int(bar_freq),
        horizon_minutes=int(getattr(constraints, "horizon_minutes", 30) or 30),
    )
    last_close = features.get("last_close")

    trading_day, intraday_today_records = _intraday_today_only(
        trade_history_records or [],
        now_ts=now_ts or _fmt_ts(snap_ts),
    )
    intraday_summary = _summarize_intraday(intraday_today_records)

    intraday_max = int(getattr(settings, "max_trades_per_day", 50) or 50)
    cost_model = _cost_model_from_constraints(constraints)

    # Optional market semantics (safe defaults if missing)
    # If you later support ETFs or other instruments allowing same-day round-trips,
    # set allow_intraday_roundtrip=True in market_contract.
    allow_intraday_roundtrip = bool(mc.get("allow_intraday_roundtrip", False))
    t_plus_one = bool(mc.get("t_plus_one", True))  # CN-A stocks default to T+1

    market_data_obj: Dict[str, Any] = {
        "bars": bars,
        "note": (
            "bars are intraday OHLCV. instrument.bar_freq_minutes tells the bar size. "
            "lookback_minutes (if provided) indicates the intended history window used to fetch these bars."
        ),
    }
    if int(bar_freq) == 5:
        market_data_obj["bars_5m"] = bars  # backward compatibility

    # Strategy-mode policy knobs: conservative hurdle, size scaling when confirmed
    strategy_mode_hurdle_multipliers = {"mild": 1.30, "normal": 1.10, "aggressive": 1.00}
    strategy_mode_size_multipliers = {"mild": 0.70, "normal": 1.00, "aggressive": 1.30}

    payload: Dict[str, Any] = {
        "task": (
            "Produce a 30-minute trading SIGNAL (BUY/SELL/HOLD) using intraday OHLCV + derived_features + fee model; "
            "size MUST be in lots (手). Final execution gating is handled by the system."
        ),
        "now": {
            "ts": now_ts or _fmt_ts(snap_ts),
            "timezone": timezone_name,
            "note": "Decision time is aligned to the latest completed bar <= now.",
        },
        "instrument": {
            "symbol": snapshot.symbol,
            "market": "CN-A",
            "bar_freq_minutes": int(bar_freq),
            "lookback_minutes": int(lookback_minutes) if lookback_minutes is not None else None,
            "currency": getattr(constraints, "currency", "CNY") or "CNY",
            "current_reference_price_close": float(last_close) if last_close is not None else None,
            "t_plus_one": t_plus_one,
            "allow_intraday_roundtrip": allow_intraday_roundtrip,
        },
        "account_state": {
            "cash_cny": float(getattr(account, "cash_cny", 0.0) or 0.0),
            "position_shares": int(getattr(account, "position_shares", 0) or 0),
            "avg_cost_cny": float(account.avg_cost_cny)
            if getattr(account, "avg_cost_cny", None) is not None
            else None,
            "unrealized_pnl_cny": float(account.unrealized_pnl_cny)
            if getattr(account, "unrealized_pnl_cny", None) is not None
            else None,
        },
        "execution_constraints": {
            "horizon_minutes": int(constraints.horizon_minutes),
            "max_trades_left_today": int(constraints.max_trades_left_today),
            "lot_size": int(constraints.lot_size),
            "max_order_shares": int(constraints.max_order_shares)
            if constraints.max_order_shares is not None
            else None,
            "min_confidence_to_trade": float(constraints.min_confidence_to_trade),
            "fees_bps_est": int(getattr(constraints, "fees_bps_est", 0) or 0),
            "slippage_bps_est": int(getattr(constraints, "slippage_bps_est", 0) or 0),
            "market_contract": mc,
            "sizing_rule": "Output suggested_lots (手). Shares = suggested_lots * lot_size. For HOLD use suggested_lots=0.",
        },
        "transaction_costs": cost_model,
        "derived_features": features,
        "market_data": market_data_obj,
        "intraday_trade_history": {
            "trading_day_shanghai": trading_day.isoformat(),
            "summary": intraday_summary,
            "records_today": intraday_today_records[-intraday_max:],
            "how_to_use": [
                "Use TODAY feedback to adjust confidence and suggested_lots.",
                "If today is losing, reduce suggested_lots and confidence unless current setup is clearly stronger and fee-positive.",
                "Avoid flip-flopping unless there is a clear regime change (breakout/breakdown/reversal).",
            ],
        },
        "fee_aware_edge_guidance": {
            "momentum_proxy_fields": ["derived_features.ret_horizon", "derived_features.ret_30m", "derived_features.ret_15m"],
            "regime_fields": [
                "derived_features.pos_in_range_0_1",
                "derived_features.realized_vol_recent",
                "derived_features.range_pct_N",
                "derived_features.volume_ratio_recent",
            ],
            "base_hurdle_field": "transaction_costs.round_trip_friction_rate_est",
            "effective_hurdle_formula": (
                "effective_hurdle = round_trip_friction_rate_est * decision_policy.strategy_mode_hurdle_multipliers[strategy_mode]"
            ),
            "rule_of_thumb": (
                "If abs(momentum_proxy) < effective_hurdle: edge likely fee-negative -> HOLD or tiny size with low confidence. "
                "If comfortably above AND confirmed by regime/volume: confidence and size can increase."
            ),
            "note": "ret_* fields are backward-looking momentum proxies; combine with regime/volume to infer next-30m drift.",
        },
        "decision_policy": {
            "goal": decision_goal,
            "trade_style": trade_style,
            "strategy_mode": strategy_mode,
            "strategy_mode_hurdle_multipliers": strategy_mode_hurdle_multipliers,
            "strategy_mode_size_multipliers": strategy_mode_size_multipliers,
            "strategy_mode_guidance": {
                "mild": (
                    "Conservative: require stronger confirmation and higher net edge. "
                    "Prefer HOLD when borderline; smaller size; avoid flips."
                ),
                "normal": (
                    "Balanced: trade when fee-positive edge clears hurdle with margin and has confirmation. "
                    "Size proportionally to confidence and net edge."
                ),
                "aggressive": (
                    "More active only when confirmed: do NOT trade merely because edge barely clears costs. "
                    "Keep cost hurdle conservative; express aggressiveness mainly through larger size when signals align."
                ),
            },
            "signal_rule": (
                "expected_direction is drift: UP/DOWN/FLAT. "
                "UP drift -> BUY. DOWN drift -> SELL (reduce existing long). "
                "FLAT drift can still BUY/SELL as mean-reversion near range extremes; otherwise HOLD."
            ),
            "position_management": "Scale in/out conservatively; avoid rapid flips within the day.",
            "constraints_handling": (
                "If constraints likely block execution (e.g., max_trades_left_today==0), "
                "keep the best direction but set suggested_lots=0 and mention the block briefly."
            ),
            "market_semantics": (
                "Assume long-only. If position_shares==0, prefer HOLD over SELL. "
                "If instrument.t_plus_one is true, avoid reasoning that requires same-day round-trip selling."
            ),
            "cost_awareness": (
                "Use transaction_costs.round_trip_friction_rate_est as conservative hurdle. "
                "Apply strategy_mode_hurdle_multipliers[strategy_mode] to form effective_hurdle. "
                "Only scale size when the edge is fee-positive with margin AND confirmed; "
                "size scaling may use strategy_mode_size_multipliers[strategy_mode]."
            ),
            "note": "Final execution gating is outside the LLM.",
        },
        "output_required": {
            "return_only_json": True,
            "horizon_minutes": 30,
            "size_unit": "lots",
            "schema_keys_exact": ["action", "horizon_minutes", "confidence", "expected_direction", "suggested_lots", "reason", "risk_notes"],
        },
        "important_rules": [
            "Output ONLY valid JSON with EXACT schema keys; no extra keys.",
            "Size MUST be suggested_lots (integer lots/手). Do not output money sizing.",
            "Keep confidence calibrated and fee-aware; require margin above costs.",
            "Avoid churn and flip-flops; use TODAY history as feedback.",
            "ret_* fields are momentum proxies (backward-looking), not forecasts; combine with regime/volume.",
        ],
    }

    payload = _drop_none(payload)

    logger.info(
        "PROMPT DATA | sym=%s bar_freq=%sm bars=%s intraday_today=%s cash=%.2f lot_size=%s lookback_min=%s rt_cost_bps=%.2f mode=%s",
        getattr(snapshot, "symbol", None),
        int(bar_freq),
        len(snapshot.recent_bars or []),
        len(intraday_today_records or []),
        float(getattr(account, "cash_cny", 0.0) or 0.0),
        int(constraints.lot_size),
        int(lookback_minutes) if lookback_minutes is not None else None,
        float(cost_model.get("round_trip_friction_bps_est") or 0.0),
        strategy_mode,
    )

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
