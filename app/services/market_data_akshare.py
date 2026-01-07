# /app/services/market_data_akshare.py
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional, List

import pandas as pd
from fastapi.concurrency import run_in_threadpool

import akshare as ak
from zoneinfo import ZoneInfo

from app.logging_config import logger
from app.models import MarketSnapshot, Bar

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
        raise ValueError(f"Invalid end_ts format: {end_ts!r} (expect 'YYYY-MM-DD HH:MM:SS' in Asia/Shanghai)") from e


class AkShareAStockProvider:
    """
    Mainland A-share (沪深京) market data via AkShare/Eastmoney.

    Supports:
      - end_ts: decision time in Asia/Shanghai ("YYYY-MM-DD HH:MM:SS")
        Used to align the bar window end time for reproducible tests.
    """

    async def get_snapshot(
        self,
        symbol: str,
        *,
        include_orderbook: bool = True,
        min_period: str = "5",            # "1","5","15","30","60"
        min_lookback_minutes: int = 300,  # fetch last N minutes (best-effort)
        end_ts: Optional[str] = None,     # ✅ NEW: Shanghai time string
    ) -> MarketSnapshot:
        code = symbol.strip()

        # 1) Real-time quote table (all A shares)
        spot_df = await run_in_threadpool(ak.stock_zh_a_spot_em)
        row = spot_df.loc[spot_df["代码"] == code]
        if row.empty:
            raise ValueError(f"symbol not found in stock_zh_a_spot_em: {code}")

        r0 = row.iloc[0].to_dict()

        last_price = _to_float(r0.get("最新价"))
        if last_price is None:
            raise ValueError(f"missing 最新价 for {code}")

        # 2) Optional: order book snapshot (5-level bid/ask)
        orderbook = None
        if include_orderbook:
            try:
                ob_df = await run_in_threadpool(ak.stock_bid_ask_em, code)
                orderbook = ob_df.to_dict(orient="records")
            except Exception:
                orderbook = None

        # 3) Minute bars (best-effort, recent window) aligned to end_ts if provided
        recent_bars: List[Bar] = []

        end_dt_cn = _parse_end_ts_cn(end_ts) or datetime.now(CN_TZ)
        start_dt_cn = end_dt_cn - timedelta(minutes=int(min_lookback_minutes))

        # AkShare expects local-market timestamps formatted like "YYYY-MM-DD HH:MM:SS"
        start_str = start_dt_cn.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_dt_cn.strftime("%Y-%m-%d %H:%M:%S")

        logger.warning(
            "AKSHARE range | symbol=%s start=%s end=%s period=%s end_ts_input=%r",
            code, start_str, end_str, min_period, end_ts
        )

        try:
            min_df = await run_in_threadpool(
                ak.stock_zh_a_hist_min_em,
                code,
                start_str,
                end_str,
                min_period,
                ""  # adjust="" for intraday
            )

            logger.warning(
                "AKSHARE min_df | symbol=%s rows=%s cols=%s",
                code,
                0 if min_df is None else len(min_df),
                None if min_df is None else list(min_df.columns),
            )

            if min_df is not None and not min_df.empty:
                logger.warning("AKSHARE sample rows:\n%s", min_df.head(3).to_string())

            if min_df is not None and not min_df.empty:
                # Some versions use "时间", some use "日期"
                tcol = "时间" if "时间" in min_df.columns else ("日期" if "日期" in min_df.columns else None)
                if tcol is not None:
                    min_df[tcol] = pd.to_datetime(min_df[tcol], errors="coerce")
                    min_df = min_df.dropna(subset=[tcol])

                    # IMPORTANT:
                    # - Treat returned timestamps as China local time by default.
                    # - Convert them to UTC for your Bar.ts (your system assumes UTC on bars).
                    # This avoids the old bug: `.replace(tzinfo=timezone.utc)` (wrong) which
                    # interprets CN timestamps as if they were already UTC.
                    dt_cn = min_df[tcol].dt.tz_localize(CN_TZ, nonexistent="shift_forward", ambiguous="NaT")
                    min_df = min_df.assign(_ts_utc=dt_cn.dt.tz_convert(timezone.utc))

                    # Guard: keep only bars <= end_dt_cn (in CN time)
                    end_dt_utc = end_dt_cn.astimezone(timezone.utc)
                    min_df = min_df[min_df["_ts_utc"] <= end_dt_utc]

                    # keep last N rows
                    min_df = min_df.tail(500)

                    for _, rr in min_df.iterrows():
                        ts_utc = rr["_ts_utc"].to_pydatetime()

                        o = _to_float(rr.get("开盘", rr.get("open", rr.get("Open")))) or 0.0
                        h = _to_float(rr.get("最高", rr.get("high", rr.get("High")))) or 0.0
                        l = _to_float(rr.get("最低", rr.get("low", rr.get("Low")))) or 0.0
                        c = _to_float(rr.get("收盘", rr.get("close", rr.get("Close", rr.get("最新价"))))) or 0.0
                        v = _to_float(rr.get("成交量", rr.get("volume", rr.get("Volume")))) or 0.0

                        # amount + vwap
                        amt = _to_float(rr.get("成交额", rr.get("amount", rr.get("Amount"))))
                        avgp = _to_float(rr.get("均价", rr.get("vwap", rr.get("VWAP", rr.get("avg_price")))))

                        recent_bars.append(
                            Bar(
                                ts=ts_utc,
                                open=float(o),
                                high=float(h),
                                low=float(l),
                                close=float(c),
                                volume=float(v),
                                amount=amt,
                                vwap=avgp,
                            )
                        )

        except Exception as e:
            logger.exception(
                "AKSHARE ERROR | symbol=%s period=%s lookback_min=%s end_ts=%r err=%s",
                code,
                min_period,
                min_lookback_minutes,
                end_ts,
                e,
            )
            recent_bars = []

        # Ensure ascending (defensive)
        recent_bars.sort(key=lambda b: b.ts)

        # 4) Build snapshot
        snapshot = MarketSnapshot(
            symbol=code,
            ts=datetime.now(timezone.utc),
            last_price=float(last_price),
            day_open=_to_float(r0.get("今开")),
            day_high=_to_float(r0.get("最高")),
            day_low=_to_float(r0.get("最低")),
            day_volume=_to_float(r0.get("成交量")),
            vwap=_to_float(r0.get("均价")) if "均价" in r0 else None,
            prev_close=_to_float(r0.get("昨收")),
            recent_bars=recent_bars,
            extra={
                "source": "akshare/eastmoney",
                "raw_spot_row": r0,
                "orderbook": orderbook,
                # Helpful debug meta:
                "bars_window_cn": {"start": start_str, "end": end_str, "period": min_period},
            },
        )
        return snapshot
