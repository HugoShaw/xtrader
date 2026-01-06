# app/services/stock_api.py
from __future__ import annotations

from typing import Literal, Dict, Any
import pandas as pd
import akshare as ak

_ALLOWED_FREQ = {"1", "5", "15", "30", "60"}


def fetch_cn_minute_bars(
    *,
    symbol: str,
    start: str,
    end: str,
    freq: Literal["1", "5", "15", "30", "60"] = "5",
    limit: int = 2000,
) -> pd.DataFrame:
    """
    Fetch A-share minute bars via AkShare/Eastmoney.
    Returns DataFrame indexed by datetime with at least: open, high, low, close, volume, amount (if available).
    """
    if freq not in _ALLOWED_FREQ:
        raise ValueError(f"Invalid freq={freq}, allowed: {sorted(_ALLOWED_FREQ)}")

    df = ak.stock_zh_a_hist_min_em(
        symbol=symbol,
        start_date=start,
        end_date=end,
        period=freq,
        adjust="",
    )
    if df is None or df.empty:
        raise ValueError("No minute data returned. Try a closer date range or larger freq.")

    tcol = "时间" if "时间" in df.columns else ("日期" if "日期" in df.columns else None)
    if tcol is None:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")

    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol).set_index(tcol)

    rename_map = {}
    for cn, en in [
        ("开盘", "open"),
        ("最高", "high"),
        ("最低", "low"),
        ("收盘", "close"),
        ("成交量", "volume"),
        ("成交额", "amount"),
    ]:
        if cn in df.columns:
            rename_map[cn] = en
    df = df.rename(columns=rename_map)

    if "close" not in df.columns and "最新价" in df.columns:
        df["close"] = pd.to_numeric(df["最新价"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"])

    if limit and len(df) > limit:
        df = df.tail(limit)

    return df


def bars_to_payload(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert to lightweight JSON payload for frontend plotting.
    """
    cols = [c for c in ["open", "high", "low", "close", "volume", "amount"] if c in df.columns]
    out = df[cols].copy()
    out = out.reset_index().rename(columns={out.index.name or "index": "ts"})
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return {"bars": out.to_dict(orient="records")}
