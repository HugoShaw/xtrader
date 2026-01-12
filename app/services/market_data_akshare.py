# /app/services/market_data_akshare.py
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi.concurrency import run_in_threadpool
from zoneinfo import ZoneInfo

import akshare as ak

from app.logging_config import logger
from app.models import MarketSnapshot, Bar
from app.services.cache import AsyncCache

CN_TZ = ZoneInfo("Asia/Shanghai")


def _to_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None


def _parse_end_ts_cn(end_ts: Optional[str]) -> Optional[datetime]:
    """
    Parse end_ts in Shanghai time: "YYYY-MM-DD HH:MM:SS" -> aware datetime in Asia/Shanghai.
    """
    if not end_ts:
        return None
    s = str(end_ts).strip()
    if not s:
        return None
    try:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=CN_TZ)
    except Exception as e:
        raise ValueError(
            f"Invalid end_ts format: {end_ts!r} (expect 'YYYY-MM-DD HH:MM:SS' in Asia/Shanghai)"
        ) from e


class AkShareAStockProvider:
    """
    Mainland A-share (沪深京) market data via AkShare/Eastmoney.

    This version:
      - DOES NOT use stock_zh_a_spot_em
      - DOES NOT use stock_intraday_em
      - Uses ONLY stock_zh_a_hist_min_em for prices + OHLCV
      - Optionally uses stock_bid_ask_em for orderbook

    Notes:
      - last_price/day_* derived from minute bars.
      - prev_close is not directly available from hist_min_em; we keep it None.
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
            return cached  # may be None or list[dict]

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
            return cached  # list[Bar] in memory cache

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

                ts = pd.to_datetime(min_df[tcol], errors="coerce")
                min_df = min_df.assign(_ts_cn=ts).dropna(subset=["_ts_cn"])
                ts_cn = min_df["_ts_cn"].dt.tz_localize(CN_TZ, nonexistent="shift_forward", ambiguous="NaT")
                min_df = min_df.assign(_ts_utc=ts_cn.dt.tz_convert(timezone.utc)).dropna(subset=["_ts_utc"])

                # keep last N rows (avoid huge memory)
                min_df = min_df.tail(800)

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

                # vwap column is sometimes present; if not, we'll compute later
                vwap_col = pd.to_numeric(min_df["均价"], errors="coerce") if "均价" in min_df.columns else None

                ts_utc_list = min_df["_ts_utc"].dt.to_pydatetime()

                bars: List[Bar] = []
                for i in range(len(min_df)):
                    bars.append(
                        Bar(
                            ts=ts_utc_list[i],
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
        min_lookback_minutes: int = 300,
        end_ts: Optional[str] = None,
    ) -> MarketSnapshot:
        code = symbol.strip()

        end_dt_cn = _parse_end_ts_cn(end_ts) or datetime.now(CN_TZ)
        start_dt_cn = end_dt_cn - timedelta(minutes=int(min_lookback_minutes))
        start_str = start_dt_cn.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_dt_cn.strftime("%Y-%m-%d %H:%M:%S")

        # concurrent fetch
        ob_task = asyncio.create_task(self._get_orderbook(code)) if include_orderbook else None
        bars_task = (
            asyncio.create_task(self._get_minute_bars(code, start_str=start_str, end_str=end_str, min_period=min_period))
            if include_bars
            else None
        )

        orderbook = await ob_task if ob_task is not None else None
        recent_bars = await bars_task if bars_task is not None else []

        # Derive snapshot fields from bars
        if recent_bars:
            last_bar = recent_bars[-1]
            last_price = float(last_bar.close)

            day_open = float(recent_bars[0].open)
            day_high = float(max(b.high for b in recent_bars))
            day_low = float(min(b.low for b in recent_bars))

            # volume/amount aggregation across window
            day_volume = float(sum(b.volume for b in recent_bars))

            # vwap:
            # prefer amount-weighted if amount exists; else fallback to volume-weighted close
            amount_sum = 0.0
            vol_sum_for_vwap = 0.0
            notional = 0.0

            for b in recent_bars:
                if b.amount is not None and b.amount > 0:
                    amount_sum += float(b.amount)
                    # if amount exists, vwap over the window = total_amount / total_volume_shares
                    # but we don't know share volume units perfectly across providers.
                    # We'll compute a "close*volume" fallback if amount is missing.
                if b.volume > 0:
                    vol_sum_for_vwap += float(b.volume)
                    notional += float(b.close) * float(b.volume)

            vwap = None
            if amount_sum > 0 and vol_sum_for_vwap > 0:
                # some providers: amount is in CNY, volume is shares -> amount/volume ~= vwap
                vwap = float(amount_sum / vol_sum_for_vwap)
            elif vol_sum_for_vwap > 0:
                vwap = float(notional / vol_sum_for_vwap)

        else:
            # No bars => cannot compute anything meaningful
            last_price = 0.0
            day_open = None
            day_high = None
            day_low = None
            day_volume = None
            vwap = None

            logger.warning(
                "AK snapshot has no bars | symbol=%s start=%s end=%s period=%s end_ts=%r",
                code,
                start_str,
                end_str,
                min_period,
                end_ts,
            )

        return MarketSnapshot(
            symbol=code,
            ts=datetime.now(timezone.utc),
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
                "derived_from": "stock_zh_a_hist_min_em_only",
            },
        )
