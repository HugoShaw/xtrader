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


def _to_python_scalar(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if isinstance(v, (int, float, str, bool)):
        return v
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    return str(v)


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

    time_col = "\u65f6\u95f4"
    date_col = "\u65e5\u671f"
    tcol = time_col if time_col in df.columns else (date_col if date_col in df.columns else None)
    if tcol is None:
        raise ValueError(f"Unexpected columns (missing {time_col}/{date_col}): {list(df.columns)}")

    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol).set_index(tcol)

    rename_map = {}
    pairs = [
        ("\u5f00\u76d8", "open"),
        ("\u6700\u9ad8", "high"),
        ("\u6700\u4f4e", "low"),
        ("\u6536\u76d8", "close"),
        ("\u6210\u4ea4\u91cf", "volume"),
        ("\u6210\u4ea4\u989d", "amount"),
        ("\u5747\u4ef7", "vwap"),
        ("\u6da8\u8dcc\u5e45", "pct_change"),
        ("\u6da8\u8dcc\u989d", "change"),
        ("\u632f\u5e45", "amplitude"),
        ("\u6362\u624b\u7387", "turnover"),
    ]
    for cn, en in pairs:
        if cn in df.columns:
            rename_map[cn] = en
    df = df.rename(columns=rename_map)

    if "close" not in df.columns and "\u6700\u65b0\u4ef7" in df.columns:
        df["close"] = pd.to_numeric(df["\u6700\u65b0\u4ef7"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume", "amount", "vwap", "pct_change", "change", "amplitude", "turnover"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"])

    if limit is not None:
        limit = int(limit)
        if limit > 0 and len(df) > limit:
            df = df.tail(limit)

    return df


def fetch_cn_daily_bars(
    *,
    symbol: str,
    start: str,
    end: str,
    adjust: str = "",
) -> pd.DataFrame:
    """
    Fetch A-share daily bars via AkShare.
    Output DataFrame indexed by date with normalized columns:
      - open, high, low, close, volume
      - amount (optional)
    """
    df = ak.stock_zh_a_hist(
        symbol=symbol.strip(),
        start_date=start,
        end_date=end,
        period="daily",
        adjust=adjust,
    )
    if df is None or df.empty:
        raise ValueError("No daily data returned. Try a closer date range.")

    date_col = "\u65e5\u671f"
    time_col = "\u65f6\u95f4"
    tcol = date_col if date_col in df.columns else (time_col if time_col in df.columns else None)
    if tcol is None:
        raise ValueError(f"Unexpected columns (missing {date_col}/{time_col}): {list(df.columns)}")

    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol).set_index(tcol)

    rename_map = {}
    pairs = [
        ("\u5f00\u76d8", "open"),
        ("\u6700\u9ad8", "high"),
        ("\u6700\u4f4e", "low"),
        ("\u6536\u76d8", "close"),
        ("\u6210\u4ea4\u91cf", "volume"),
        ("\u6210\u4ea4\u989d", "amount"),
        ("\u632f\u5e45", "amplitude"),
        ("\u6da8\u8dcc\u5e45", "pct_change"),
        ("\u6da8\u8dcc\u989d", "change"),
        ("\u6362\u624b\u7387", "turnover"),
    ]
    for cn, en in pairs:
        if cn in df.columns:
            rename_map[cn] = en
    df = df.rename(columns=rename_map)

    if "close" not in df.columns:
        raise ValueError(f"Unexpected columns (missing close): {list(df.columns)}")

    for col in ["open", "high", "low", "close", "volume", "amount", "amplitude", "pct_change", "change", "turnover"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"])
    return df


def bars_to_payload(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert DataFrame to lightweight JSON payload for frontend plotting.
    Keeps keys aligned with Bar model / prompt usage:
      open/high/low/close/volume/amount/vwap
    """
    cols = [
        c
        for c in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "vwap",
            "amplitude",
            "pct_change",
            "change",
            "turnover",
        ]
        if c in df.columns
    ]
    out = df[cols].copy()

    out = out.reset_index().rename(columns={out.index.name or "index": "ts"})
    out["ts"] = pd.to_datetime(out["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    for c in cols:
        out[c] = out[c].apply(lambda x: None if pd.isna(x) else float(x))

    return {"bars": out.to_dict(orient="records")}


def fetch_cn_period_bars(
    *,
    symbol: str,
    start: str,
    end: str,
    period: Literal["daily", "weekly", "monthly"] = "daily",
    adjust: str = "",
) -> pd.DataFrame:
    """
    Fetch A-share bars via AkShare with a native period.
    Output DataFrame indexed by date with normalized columns:
      - open, high, low, close, volume
      - amount (optional)
    """
    df = ak.stock_zh_a_hist(
        symbol=symbol.strip(),
        start_date=start,
        end_date=end,
        period=period,
        adjust=adjust,
    )
    if df is None or df.empty:
        raise ValueError(f"No {period} data returned. Try a closer date range.")

    date_col = "\u65e5\u671f"
    time_col = "\u65f6\u95f4"
    tcol = date_col if date_col in df.columns else (time_col if time_col in df.columns else None)
    if tcol is None:
        raise ValueError(f"Unexpected columns (missing {date_col}/{time_col}): {list(df.columns)}")

    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol).set_index(tcol)

    rename_map = {}
    pairs = [
        ("\u5f00\u76d8", "open"),
        ("\u6700\u9ad8", "high"),
        ("\u6700\u4f4e", "low"),
        ("\u6536\u76d8", "close"),
        ("\u6210\u4ea4\u91cf", "volume"),
        ("\u6210\u4ea4\u989d", "amount"),
        ("\u632f\u5e45", "amplitude"),
        ("\u6da8\u8dcc\u5e45", "pct_change"),
        ("\u6da8\u8dcc\u989d", "change"),
        ("\u6362\u624b\u7387", "turnover"),
    ]
    for cn, en in pairs:
        if cn in df.columns:
            rename_map[cn] = en
    df = df.rename(columns=rename_map)

    if "close" not in df.columns:
        raise ValueError(f"Unexpected columns (missing close): {list(df.columns)}")

    for col in ["open", "high", "low", "close", "volume", "amount", "amplitude", "pct_change", "change", "turnover"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"])
    return df


def fetch_cn_stock_basic_info(*, symbol: str) -> Dict[str, Any]:
    """
    Fetch A-share basic info via AkShare/Eastmoney.
    Returns a dict keyed by 'item' with normalized python scalars.
    """
    df = ak.stock_individual_info_em(symbol=symbol.strip())
    if df is None or df.empty:
        raise ValueError("No stock basic info returned.")

    item_col = "item" if "item" in df.columns else None
    value_col = "value" if "value" in df.columns else None
    if item_col is None or value_col is None:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")

    out: Dict[str, Any] = {}
    for _, row in df.iterrows():
        key = str(row[item_col]).strip()
        val = _to_python_scalar(row[value_col])
        out[key] = val
    return out
