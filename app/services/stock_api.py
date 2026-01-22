# app/services/stock_api.py
from __future__ import annotations

from typing import Literal, Dict, Any, Optional
import pandas as pd
import akshare as ak

_ALLOWED_FREQ = {"1", "5", "15", "30", "60"}


def _to_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None


def fetch_cn_minute_bars(
    *,
    symbol: str,
    start: str,
    end: str,
    freq: Literal["1", "5", "15", "30", "60"] = "1",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch A-share minute bars via AkShare/Eastmoney.

    Output DataFrame indexed by datetime, with normalized columns:
      - open, high, low, close, volume
      - amount (成交额) (optional but usually available)
      - vwap   (均价)   (optional but usually available)

    Notes:
    - AkShare intraday range is best-effort and upstream-limited.
    - This function normalizes CN column names to EN keys for stable downstream usage.
    """
    if freq not in _ALLOWED_FREQ:
        raise ValueError(f"Invalid freq={freq}, allowed: {sorted(_ALLOWED_FREQ)}")

    df = ak.stock_zh_a_hist_min_em(
        symbol=symbol.strip(),
        start_date=start,
        end_date=end,
        period=freq,
        adjust="",
    )
    if df is None or df.empty:
        raise ValueError("No minute data returned. Try a closer date range or larger freq.")

    # Identify timestamp column
    tcol = "时间" if "时间" in df.columns else ("日期" if "日期" in df.columns else None)
    if tcol is None:
        raise ValueError(f"Unexpected columns (missing 时间/日期): {list(df.columns)}")

    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol).set_index(tcol)

    # Normalize columns (CN first, then possible EN fallbacks)
    rename_map = {}
    pairs = [
        ("开盘", "open"),
        ("最高", "high"),
        ("最低", "low"),
        ("收盘", "close"),
        ("成交量", "volume"),
        ("成交额", "amount"),  # NEW keep
        ("均价", "vwap"),      # NEW keep
    ]
    for cn, en in pairs:
        if cn in df.columns:
            rename_map[cn] = en
    df = df.rename(columns=rename_map)

    # Some versions might not have 收盘 but have 最新价
    if "close" not in df.columns and "最新价" in df.columns:
        df["close"] = pd.to_numeric(df["最新价"], errors="coerce")

    # Coerce numeric
    for col in ["open", "high", "low", "close", "volume", "amount", "vwap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["close"])

    # Limit rows from the end
    if limit is not None:
        limit = int(limit)
        if limit > 0 and len(df) > limit:
            df = df.tail(limit)

    return df


def bars_to_payload(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert DataFrame to lightweight JSON payload for frontend plotting.
    Keeps keys aligned with Bar model / prompt usage:
      open/high/low/close/volume/amount/vwap
    """
    cols = [c for c in ["open", "high", "low", "close", "volume", "amount", "vwap"] if c in df.columns]
    out = df[cols].copy()

    out = out.reset_index().rename(columns={out.index.name or "index": "ts"})
    out["ts"] = pd.to_datetime(out["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Ensure JSON-safe floats (optional; pandas will usually do fine)
    for c in cols:
        out[c] = out[c].apply(lambda x: None if pd.isna(x) else float(x))

    return {"bars": out.to_dict(orient="records")}
