# app/services/market_data_akshare.py
from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi.concurrency import run_in_threadpool

import akshare as ak

from app.logging_config import logger
from app.models import MarketSnapshot, Bar
from app.services.cache import AsyncCache
from app.utils.timeutils import (
    now_shanghai,
    must_parse_shanghai,
    fmt_shanghai,
    ensure_shanghai,
    tz,
)

CN_TZ = tz("Asia/Shanghai")


def _parse_end_ts_cn(end_ts: Optional[str]) -> Optional[pd.Timestamp]:
    """
    Parse end_ts in Shanghai time: "YYYY-MM-DD HH:MM:SS" -> tz-aware datetime in Asia/Shanghai.
    """
    if end_ts is None:
        return None
    s = str(end_ts).strip()
    if not s:
        return None
    try:
        # returns tz-aware datetime
        dt = must_parse_shanghai(s)
        # pandas Timestamp helps downstream formatting/parsing if needed
        return pd.Timestamp(dt)
    except Exception as e:
        raise ValueError(
            f"Invalid end_ts format: {end_ts!r} (expect 'YYYY-MM-DD HH:MM:SS' in Asia/Shanghai)"
        ) from e

def _day_bounds_shanghai(end_dt_cn: pd.Timestamp, *, session_only: bool = False) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Given a tz-aware Shanghai timestamp, return [start_of_day, end_dt_cn].
    If session_only=True, use CN-A regular session start 09:30:00 instead of 00:00:00.
    """
    end_dt_cn = pd.Timestamp(end_dt_cn).tz_convert(CN_TZ) if end_dt_cn.tzinfo else pd.Timestamp(end_dt_cn).tz_localize(CN_TZ)

    d = end_dt_cn.date()
    if session_only:
        start = pd.Timestamp(f"{d} 09:30:00").tz_localize(CN_TZ)
    else:
        start = pd.Timestamp(f"{d} 00:00:00").tz_localize(CN_TZ)

    # clamp: start cannot be after end
    if start > end_dt_cn:
        start = end_dt_cn

    return start, end_dt_cn


class AkShareAStockProvider:
    """
    Mainland A-share (沪深京) market data via AkShare/Eastmoney.

    Uses ONLY stock_zh_a_hist_min_em for prices + OHLCV
    Optional stock_bid_ask_em for orderbook
    """

    def __init__(
        self,
        *,
        cache: AsyncCache,
        orderbook_ttl_sec: float = 1.0,
        bars_ttl_sec: float = 2.0,
    ) -> None:
        self.cache = cache
        self.orderbook_ttl_sec = float(orderbook_ttl_sec)
        self.bars_ttl_sec = float(bars_ttl_sec)
        self._key_locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, key: str) -> asyncio.Lock:
        lk = self._key_locks.get(key)
        if lk is None:
            lk = asyncio.Lock()
            self._key_locks[key] = lk
        return lk

    # -----------------------------
    # Orderbook (optional)
    # -----------------------------
    async def _get_orderbook(self, code: str) -> Optional[List[Dict[str, Any]]]:
        ck = f"ak:ob:{code}"
        cached = await self.cache.get(ck)
        if cached is not None:
            return cached

        lk = self._lock_for(ck)
        async with lk:
            cached = await self.cache.get(ck)
            if cached is not None:
                return cached
            try:
                t0 = time.perf_counter()
                ob_df = await run_in_threadpool(ak.stock_bid_ask_em, code)
                ms = (time.perf_counter() - t0) * 1000.0
                val = ob_df.to_dict(orient="records")
                await self.cache.set(ck, val, ttl_sec=self.orderbook_ttl_sec)
                logger.info("AK orderbook | symbol=%s ms=%.1f ttl=%.1fs", code, ms, self.orderbook_ttl_sec)
                return val
            except Exception as e:
                logger.warning("AK orderbook failed | symbol=%s err=%s: %s", code, type(e).__name__, e)
                await self.cache.set(ck, None, ttl_sec=self.orderbook_ttl_sec)
                return None

    # -----------------------------
    # Minute bars (structured OHLC)
    # -----------------------------
    async def _get_minute_bars(
        self,
        code: str,
        *,
        start_str: str,
        end_str: str,
        min_period: str,
    ) -> List[Bar]:
        ck = f"ak:bars:{code}:{start_str}:{end_str}:{min_period}"
        cached = await self.cache.get(ck)
        if cached is not None:
            return cached

        lk = self._lock_for(ck)
        async with lk:
            cached = await self.cache.get(ck)
            if cached is not None:
                return cached

            try:
                t0 = time.perf_counter()
                min_df = await run_in_threadpool(
                    ak.stock_zh_a_hist_min_em,
                    code,
                    start_str,
                    end_str,
                    min_period,
                    "",  # adjust="" for intraday
                )
                ms = (time.perf_counter() - t0) * 1000.0

                if min_df is None or min_df.empty:
                    await self.cache.set(ck, [], ttl_sec=self.bars_ttl_sec)
                    logger.info("AK bars empty | symbol=%s ms=%.1f ttl=%.1fs", code, ms, self.bars_ttl_sec)
                    return []

                tcol = "时间" if "时间" in min_df.columns else ("日期" if "日期" in min_df.columns else None)
                if tcol is None:
                    await self.cache.set(ck, [], ttl_sec=self.bars_ttl_sec)
                    return []

                # Parse timestamps; assume they are Shanghai local (naive) and localize to Asia/Shanghai
                ts_naive = pd.to_datetime(min_df[tcol], errors="coerce")
                min_df = min_df.assign(_ts=ts_naive).dropna(subset=["_ts"])

                # Localize to Shanghai tz-aware
                # For Asia/Shanghai there is no DST ambiguity in modern data, but keep safe defaults.
                ts_cn = min_df["_ts"].dt.tz_localize(CN_TZ, nonexistent="shift_forward", ambiguous="infer")
                min_df = min_df.assign(_ts_cn=ts_cn).dropna(subset=["_ts_cn"]) 

                def num(col: str, default: float = 0.0) -> pd.Series:
                    if col not in min_df.columns:
                        return pd.Series([default] * len(min_df), index=min_df.index, dtype="float64")
                    return pd.to_numeric(min_df[col], errors="coerce").fillna(default).astype("float64")

                o = num("开盘")
                h = num("最高")
                l = num("最低")
                c = num("收盘")
                v = num("成交量")

                amt = pd.to_numeric(min_df["成交额"], errors="coerce") if "成交额" in min_df.columns else None
                vwap_col = pd.to_numeric(min_df["均价"], errors="coerce") if "均价" in min_df.columns else None

                ts_cn_list = min_df["_ts_cn"].dt.to_pydatetime()

                bars: List[Bar] = []
                for i in range(len(min_df)):
                    # ts_cn_list is already tz-aware Shanghai; ensure_shanghai is idempotent.
                    bars.append(
                        Bar(
                            ts=ensure_shanghai(ts_cn_list[i]),
                            open=float(o.iat[i]),
                            high=float(h.iat[i]),
                            low=float(l.iat[i]),
                            close=float(c.iat[i]),
                            volume=float(v.iat[i]),
                            amount=None if amt is None or pd.isna(amt.iat[i]) else float(amt.iat[i]),
                            vwap=None if vwap_col is None or pd.isna(vwap_col.iat[i]) else float(vwap_col.iat[i]),
                        )
                    )

                bars.sort(key=lambda b: b.ts)
                await self.cache.set(ck, bars, ttl_sec=self.bars_ttl_sec)
                logger.info("AK bars | symbol=%s rows=%s ms=%.1f ttl=%.1fs", code, len(bars), ms, self.bars_ttl_sec)
                return bars

            except Exception as e:
                logger.warning("AK bars failed | symbol=%s err=%s: %s", code, type(e).__name__, e)
                await self.cache.set(ck, [], ttl_sec=self.bars_ttl_sec)
                return []

    # -----------------------------
    # Snapshot
    # -----------------------------
    async def get_snapshot(
        self,
        symbol: str,
        *,
        include_orderbook: bool = False,
        include_bars: bool = True,
        min_period: str = "5",
        min_lookback_minutes: int = 300,  # kept for backward compat (unused when intraday_full_day=True)
        end_ts: Optional[str] = None,
        intraday_full_day: bool = True,   # NEW: default fetch all bars in the Shanghai trading day
        session_only: bool = True,       # NEW: if True, start at 09:30:00 instead of 00:00:00
    ) -> MarketSnapshot:
        code = symbol.strip()

        # end anchor (Shanghai tz-aware)
        end_dt_cn = _parse_end_ts_cn(end_ts)
        if end_dt_cn is None:
            end_dt_cn = pd.Timestamp(now_shanghai())
        # ensure tz-aware Shanghai
        end_dt_cn = pd.Timestamp(ensure_shanghai(end_dt_cn.to_pydatetime()))

        if intraday_full_day:
            start_dt_cn, end_dt_cn = _day_bounds_shanghai(end_dt_cn, session_only=session_only)
        else:
            # fallback to rolling window (old behavior)
            start_dt_cn = end_dt_cn - pd.Timedelta(minutes=int(min_lookback_minutes))

        start_str = fmt_shanghai(start_dt_cn.to_pydatetime()) or start_dt_cn.strftime("%Y-%m-%d %H:%M:%S")
        end_str = fmt_shanghai(end_dt_cn.to_pydatetime()) or end_dt_cn.strftime("%Y-%m-%d %H:%M:%S")

        ob_task = asyncio.create_task(self._get_orderbook(code)) if include_orderbook else None
        bars_task = (
            asyncio.create_task(self._get_minute_bars(code, start_str=start_str, end_str=end_str, min_period=min_period))
            if include_bars
            else None
        )

        orderbook = await ob_task if ob_task is not None else None
        recent_bars = await bars_task if bars_task is not None else []

        # If intraday_full_day=True, do NOT tail(800) inside _get_minute_bars.
        # Otherwise you won't actually get the full day. See note below.

        if recent_bars:
            last_bar = recent_bars[-1]
            last_price = float(last_bar.close)

            day_open = float(recent_bars[0].open)
            day_high = float(max(b.high for b in recent_bars))
            day_low = float(min(b.low for b in recent_bars))
            day_volume = float(sum(b.volume for b in recent_bars))

            amount_sum = 0.0
            vol_sum_for_vwap = 0.0
            notional = 0.0

            for b in recent_bars:
                if b.amount is not None and b.amount > 0:
                    amount_sum += float(b.amount)
                if b.volume > 0:
                    vol_sum_for_vwap += float(b.volume)
                    notional += float(b.close) * float(b.volume)

            vwap = None
            if amount_sum > 0 and vol_sum_for_vwap > 0:
                vwap = float(amount_sum / vol_sum_for_vwap)
            elif vol_sum_for_vwap > 0:
                vwap = float(notional / vol_sum_for_vwap)

            snap_ts = ensure_shanghai(last_bar.ts)
        else:
            last_price = 0.0
            day_open = None
            day_high = None
            day_low = None
            day_volume = None
            vwap = None

            logger.warning(
                "AK snapshot has no bars | symbol=%s start=%s end=%s period=%s end_ts=%r full_day=%s session_only=%s",
                code,
                start_str,
                end_str,
                min_period,
                end_ts,
                intraday_full_day,
                session_only,
            )
            snap_ts = ensure_shanghai(end_dt_cn.to_pydatetime())

        return MarketSnapshot(
            symbol=code,
            ts=snap_ts,
            last_price=float(last_price),
            day_open=day_open,
            day_high=day_high,
            day_low=day_low,
            day_volume=day_volume,
            vwap=vwap,
            prev_close=None,
            recent_bars=recent_bars,
            extra={
                "source": "akshare/eastmoney",
                "orderbook": orderbook,
                "bars_window_cn": {"start": start_str, "end": end_str, "period": min_period},
                "intraday_full_day": intraday_full_day,
                "session_only": session_only,
                "derived_from": "stock_zh_a_hist_min_em_only",
            },
        )
