from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional, List

import pandas as pd
from fastapi.concurrency import run_in_threadpool

import akshare as ak

from app.models import MarketSnapshot, Bar


def _to_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None


class AkShareAStockProvider:
    """
    Mainland A-share (沪深京) market data via AkShare/Eastmoney.

    Key interfaces used:
    - ak.stock_zh_a_spot_em()   : all A-share real-time quotes (snapshot table) :contentReference[oaicite:3]{index=3}
    - ak.stock_bid_ask_em()     : 5-level order book snapshot (symbol: 6-digit) :contentReference[oaicite:4]{index=4}
    - ak.stock_zh_a_hist_min_em(): minute intraday bars (recent only; upstream-limited) :contentReference[oaicite:5]{index=5}
    """

    async def get_snapshot(
        self,
        symbol: str,
        *,
        include_orderbook: bool = True,
        min_period: str = "5",          # "1","5","15","30","60"
        min_lookback_minutes: int = 300 # fetch last N minutes (best-effort)
    ) -> MarketSnapshot:
        code = symbol.strip()

        # 1) Real-time quote table (all A shares)
        spot_df = await run_in_threadpool(ak.stock_zh_a_spot_em)
        row = spot_df.loc[spot_df["代码"] == code]
        if row.empty:
            raise ValueError(f"symbol not found in stock_zh_a_spot_em: {code}")

        r0 = row.iloc[0].to_dict()

        # Common fields in this table include 最新价, 成交量, 成交额, 振幅, 涨跌幅... (varies with AkShare version)
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

        # 3) Minute bars (best-effort, recent window)
        recent_bars: List[Bar] = []
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(minutes=min_lookback_minutes)

        # AkShare expects local-market timestamps formatted like "YYYY-MM-DD HH:MM:SS"
        # We pass naive strings; you can convert to Asia/Shanghai precisely if you want.
        start_str = start_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        try:
            min_df = await run_in_threadpool(
                ak.stock_zh_a_hist_min_em,
                code,
                start_str,
                end_str,
                min_period,
                ""  # adjust="" for intraday; upstream-limited :contentReference[oaicite:6]{index=6}
            )
            # Typical columns: 时间/开盘/收盘/最高/最低/成交量/成交额 (may differ)
            # Normalize
            if not min_df.empty:
                # Some versions use "时间", some use "日期"
                tcol = "时间" if "时间" in min_df.columns else ("日期" if "日期" in min_df.columns else None)
                if tcol:
                    min_df[tcol] = pd.to_datetime(min_df[tcol])
                    for _, rr in min_df.tail(500).iterrows():
                        recent_bars.append(
                            Bar(
                                ts=rr[tcol].to_pydatetime().replace(tzinfo=timezone.utc),
                                open=float(rr.get("开盘", rr.get("open", rr.get("Open", 0))) or 0),
                                high=float(rr.get("最高", rr.get("high", rr.get("High", 0))) or 0),
                                low=float(rr.get("最低", rr.get("low", rr.get("Low", 0))) or 0),
                                close=float(rr.get("收盘", rr.get("最新价", rr.get("close", rr.get("Close", 0)))) or 0),
                                volume=float(rr.get("成交量", rr.get("volume", rr.get("Volume", 0))) or 0),
                            )
                        )
        except Exception:
            # Minute feed is “best-effort”; upstream can limit ranges :contentReference[oaicite:7]{index=7}
            recent_bars = []

        # 4) Build snapshot
        snapshot = MarketSnapshot(
            symbol=code,
            ts=datetime.now(timezone.utc),
            last_price=last_price,
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
            },
        )
        return snapshot
