from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import akshare as ak

from app.logging_config import logger
from app.services.stock_api import fetch_cn_period_bars, fetch_cn_minute_bars
from app.utils.timeutils import now_shanghai, fmt_shanghai


@dataclass(frozen=True)
class TimeframePlan:
    timeframe: str
    action_bias: str
    confidence: float
    summary: str
    last_close_cny: Optional[float]
    change_pct: Optional[float]
    short_ma: Optional[float]
    long_ma: Optional[float]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except Exception:
        return None


def _timeframe_windows(timeframe: str) -> Tuple[int, int]:
    if timeframe == "daily":
        return 20, 60
    if timeframe == "weekly":
        return 4, 12
    if timeframe == "monthly":
        return 3, 6
    if timeframe == "quarterly":
        return 2, 4
    return 5, 20


def _score_to_bias(score: int) -> Tuple[str, float]:
    if score >= 2:
        return "bullish", min(1.0, abs(score) / 3.0)
    if score <= -2:
        return "bearish", min(1.0, abs(score) / 3.0)
    return "neutral", min(0.6, abs(score) / 3.0)


def build_timeframe_plan(df: pd.DataFrame, timeframe: str) -> TimeframePlan:
    short_win, long_win = _timeframe_windows(timeframe)
    closes = df["close"].dropna() if df is not None and "close" in df.columns else pd.Series(dtype=float)

    if len(closes) < max(short_win, long_win) + 2:
        last_close = _safe_float(closes.iloc[-1]) if len(closes) else None
        return TimeframePlan(
            timeframe=timeframe,
            action_bias="neutral",
            confidence=0.2,
            summary="Not enough data for a confident trend read.",
            last_close_cny=last_close,
            change_pct=None,
            short_ma=None,
            long_ma=None,
        )

    short_ma = _safe_float(closes.rolling(short_win).mean().iloc[-1])
    long_ma = _safe_float(closes.rolling(long_win).mean().iloc[-1])
    last_close = _safe_float(closes.iloc[-1])
    prev_close = _safe_float(closes.iloc[-2])
    change_pct = None
    if last_close is not None and prev_close and prev_close != 0:
        change_pct = (last_close / prev_close - 1) * 100.0

    score = 0
    if short_ma is not None and long_ma is not None:
        score += 1 if short_ma > long_ma else (-1 if short_ma < long_ma else 0)
    if change_pct is not None:
        score += 1 if change_pct > 0 else (-1 if change_pct < 0 else 0)
    if last_close is not None and short_ma is not None:
        score += 1 if last_close > short_ma else (-1 if last_close < short_ma else 0)

    bias, confidence = _score_to_bias(score)

    summary_parts = []
    if short_ma is not None and long_ma is not None:
        summary_parts.append(f"MA{short_win} {short_ma:.2f} vs MA{long_win} {long_ma:.2f}")
    if change_pct is not None:
        summary_parts.append(f"last change {change_pct:.2f}%")
    summary = ", ".join(summary_parts) if summary_parts else "Trend snapshot unavailable."

    return TimeframePlan(
        timeframe=timeframe,
        action_bias=bias,
        confidence=confidence,
        summary=summary,
        last_close_cny=last_close,
        change_pct=change_pct,
        short_ma=short_ma,
        long_ma=long_ma,
    )


def combine_action(
    plans: List[TimeframePlan],
    *,
    shares: int,
    cash_cny: float,
    lot_size: int,
    last_price_cny: Optional[float],
    avg_cost_cny: Optional[float],
    unrealized_pnl_cny: Optional[float],
) -> Tuple[str, str]:
    score = 0
    for plan in plans:
        if plan.action_bias == "bullish":
            score += 1
        elif plan.action_bias == "bearish":
            score -= 1

    reasons: List[str] = []
    if shares > 0 and last_price_cny and avg_cost_cny and avg_cost_cny > 0:
        pnl_pct = (float(last_price_cny) - float(avg_cost_cny)) / float(avg_cost_cny) * 100.0
        if pnl_pct <= -5.0:
            score -= 1
            reasons.append(f"position down {pnl_pct:.1f}% vs avg cost")
        elif pnl_pct >= 5.0:
            score += 1
            reasons.append(f"position up {pnl_pct:.1f}% vs avg cost")
    if unrealized_pnl_cny is not None:
        if float(unrealized_pnl_cny) <= -1000.0:
            score -= 1
            reasons.append("unrealized PnL negative")
        elif float(unrealized_pnl_cny) >= 1000.0:
            score += 1
            reasons.append("unrealized PnL positive")

    desired = "HOLD"
    if score >= 2:
        desired = "BUY"
    elif score <= -2:
        desired = "SELL"

    if desired == "BUY":
        if not last_price_cny or last_price_cny <= 0:
            return "HOLD", "No realtime price available; skipping trade."
        min_cash = float(lot_size) * float(last_price_cny)
        if cash_cny < min_cash:
            return "HOLD", "Insufficient cash for one lot."
        base = "Multi-timeframe trend alignment is bullish."
        return "BUY", base + (f" ({'; '.join(reasons)})" if reasons else "")

    if desired == "SELL":
        if shares < int(lot_size):
            return "HOLD", "No shares available to sell."
        base = "Multi-timeframe trend alignment is bearish."
        return "SELL", base + (f" ({'; '.join(reasons)})" if reasons else "")

    base = "Signals are mixed; holding is safer."
    return "HOLD", base + (f" ({'; '.join(reasons)})" if reasons else "")


def suggest_order_shares(
    *,
    action: str,
    shares: int,
    cash_cny: float,
    lot_size: int,
    last_price_cny: Optional[float],
    order_lots: int,
) -> int:
    if action not in ("BUY", "SELL"):
        return 0
    lots = max(int(order_lots), 1)
    lot_size = max(int(lot_size), 1)

    if action == "SELL":
        max_lots = shares // lot_size
        return lot_size * min(lots, max_lots) if max_lots > 0 else 0

    if not last_price_cny or last_price_cny <= 0:
        return 0
    max_lots = int(cash_cny // (float(last_price_cny) * lot_size))
    return lot_size * min(lots, max_lots) if max_lots > 0 else 0


def fetch_realtime_quote(symbol: str) -> Optional[Dict[str, Any]]:
    t0 = time.perf_counter()
    now = now_shanghai()
    end_str = fmt_shanghai(now)
    start_str = fmt_shanghai(now - timedelta(days=2))
    if not start_str or not end_str:
        return None
    try:
        df = fetch_cn_minute_bars(
            symbol=symbol,
            start=start_str,
            end=end_str,
            freq="1",
            limit=1,
        )
    except Exception as exc:
        logger.warning("realtime_quote_error | symbol=%s err=%s", symbol, exc)
        return None

    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    ts = df.index[-1]
    payload = {
        "ts": pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S"),
        "last_price_cny": _safe_float(last.get("close")),
        "amount": _safe_float(last.get("amount")),
        "volume": _safe_float(last.get("volume")),
    }
    logger.debug("realtime_quote_ok | symbol=%s ms=%.1f", symbol, (time.perf_counter() - t0) * 1000.0)
    return payload


def fetch_news(symbol: str, limit: int = 8) -> List[Dict[str, Any]]:
    t0 = time.perf_counter()
    try:
        df = ak.stock_news_em(symbol=str(symbol))
    except Exception as exc:
        logger.warning("news_fetch_error | symbol=%s err=%s", symbol, exc)
        return []

    if df is None or df.empty:
        return []

    df = df.head(int(limit))
    items: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        items.append(
            {
                "title": str(row.get("新闻标题", "")) or "Untitled",
                "published_at": str(row.get("发布时间", "")) or None,
                "source": str(row.get("文章来源", "")) or None,
                "url": str(row.get("新闻链接", "")) or None,
                "keywords": str(row.get("关键词", "")) or None,
            }
        )
    logger.debug("news_fetch_ok | symbol=%s items=%s ms=%.1f", symbol, len(items), (time.perf_counter() - t0) * 1000.0)
    return items


def build_symbol_plans(
    *,
    symbol: str,
    start_day: str,
    end_day: str,
    adjust: str,
) -> Tuple[List[TimeframePlan], Optional[float]]:
    t0 = time.perf_counter()
    def _fetch_period(period: str) -> pd.DataFrame:
        try:
            return fetch_cn_period_bars(
                symbol=symbol,
                start=start_day,
                end=end_day,
                period=period,  # type: ignore[arg-type]
                adjust=adjust,
            )
        except Exception as exc:
            if period == "quarterly":
                logger.info(
                    "period_not_supported | symbol=%s period=%s err=%s",
                    symbol,
                    period,
                    exc,
                )
            else:
                logger.warning("period_fetch_error | symbol=%s period=%s err=%s", symbol, period, exc)
            return pd.DataFrame()

    daily_df = _fetch_period("daily")
    daily_df = daily_df.dropna(subset=["close"]) if daily_df is not None else pd.DataFrame()

    plans: List[TimeframePlan] = []
    if daily_df is None or daily_df.empty:
        logger.debug("symbol_plan_empty | symbol=%s ms=%.1f", symbol, (time.perf_counter() - t0) * 1000.0)
        return plans, None

    plans.append(build_timeframe_plan(daily_df, "daily"))
    weekly_df = _fetch_period("weekly")
    monthly_df = _fetch_period("monthly")
    quarterly_df = _fetch_period("quarterly")

    if not weekly_df.empty:
        plans.append(build_timeframe_plan(weekly_df, "weekly"))
    if not monthly_df.empty:
        plans.append(build_timeframe_plan(monthly_df, "monthly"))
    if not quarterly_df.empty:
        plans.append(build_timeframe_plan(quarterly_df, "quarterly"))

    last_close = _safe_float(daily_df["close"].iloc[-1])
    logger.debug(
        "symbol_plan_ok | symbol=%s daily=%s weekly=%s monthly=%s quarterly=%s ms=%.1f",
        symbol,
        len(daily_df),
        len(weekly_df),
        len(monthly_df),
        len(quarterly_df),
        (time.perf_counter() - t0) * 1000.0,
    )
    return plans, last_close


def llm_decision_schema_hint() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["action", "suggested_lots", "confidence", "reason"],
        "properties": {
            "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
            "suggested_lots": {"type": "integer", "minimum": 0, "maximum": 50},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string"},
            "risk_notes": {"type": "string"},
        },
        "additionalProperties": False,
    }
