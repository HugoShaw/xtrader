# app/services/stock_api.py
from __future__ import annotations

from typing import Literal, Dict, Any, Optional
import threading
import numpy as np
import pandas as pd
import akshare as ak

from app.config import settings

_ALLOWED_FREQ = {"1", "5", "15", "30", "60"}

_provider_counts: Dict[str, int] = {}
_provider_counts_lock = threading.Lock()


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


def _to_market_symbol(code: str) -> str:
    s = code.strip().lower()
    if len(s) != 6 or not s.isdigit():
        return code.strip()
    if s.startswith("6"):
        return f"sh{s}"
    if s.startswith(("0", "3")):
        return f"sz{s}"
    if s.startswith(("8", "4", "9")):
        return f"bj{s}"
    return f"sz{s}"


def _normalize_hist_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("No data returned.")

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
    df = _ensure_daily_metrics(df)
    if df.empty:
        raise ValueError("No data after normalization.")
    return df


def _normalize_sina_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("No data returned.")

    date_col = "date"
    if date_col not in df.columns:
        raise ValueError(f"Unexpected columns (missing {date_col}): {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    for col in ["open", "high", "low", "close", "volume", "amount", "outstanding_share", "turnover"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "close" not in df.columns:
        raise ValueError(f"Unexpected columns (missing close): {list(df.columns)}")

    df = df.dropna(subset=["close"])
    df = _ensure_daily_metrics(df)
    if df.empty:
        raise ValueError("No data after normalization.")
    return df


def _normalize_tx_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("No data returned.")

    date_col = "date"
    if date_col not in df.columns:
        raise ValueError(f"Unexpected columns (missing {date_col}): {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    rename_map = {}
    if "amount" in df.columns:
        rename_map["amount"] = "volume"
    df = df.rename(columns=rename_map)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "close" not in df.columns:
        raise ValueError(f"Unexpected columns (missing close): {list(df.columns)}")

    df = df.dropna(subset=["close"])
    df = _ensure_daily_metrics(df)
    if df.empty:
        raise ValueError("No data after normalization.")
    return df


def _ensure_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "close" not in df.columns:
        return df
    if "pct_change" not in df.columns:
        df["pct_change"] = df["close"].pct_change() * 100.0
    if "amplitude" not in df.columns and "high" in df.columns and "low" in df.columns:
        df["amplitude"] = (df["high"] - df["low"]) / df["close"] * 100.0
    if "change" not in df.columns:
        df["change"] = df["close"].diff()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def _resample_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("No data to resample.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected datetime index for resample.")

    period_key = str(period or "daily").lower()
    if period_key == "weekly":
        rule = "W-FRI"
    elif period_key == "monthly":
        rule = "M"
    elif period_key == "daily":
        return df
    else:
        raise ValueError(f"Unsupported resample period: {period}")

    agg_map = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "amount": "sum",
    }

    cols = {k: v for k, v in agg_map.items() if k in df.columns}
    out = df.resample(rule).agg(cols)
    out = out.dropna(subset=["close"])

    out = _ensure_daily_metrics(out)

    return out


def _record_provider_use(
    *,
    provider: str,
    symbol: str,
    start: str,
    end: str,
    period: Optional[str] = None,
    adjust: str = "",
    is_fallback: bool = False,
) -> None:
    if not settings.stock_api_fallback_log:
        return
    key = f"{provider}|fallback={1 if is_fallback else 0}"
    with _provider_counts_lock:
        _provider_counts[key] = _provider_counts.get(key, 0) + 1


def get_provider_usage_counts() -> Dict[str, int]:
    with _provider_counts_lock:
        return dict(_provider_counts)


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
    errors: list[str] = []

    def _attempt(name: str, fn) -> Optional[pd.DataFrame]:
        try:
            return fn()
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            return None

    df = _attempt(
        "stock_zh_a_hist",
        lambda: _normalize_hist_df(
            ak.stock_zh_a_hist(
                symbol=symbol.strip(),
                start_date=start,
                end_date=end,
                period="daily",
                adjust=adjust,
            )
        ),
    )
    if df is not None:
        _record_provider_use(
            provider="stock_zh_a_hist",
            symbol=symbol,
            start=start,
            end=end,
            period="daily",
            adjust=adjust,
            is_fallback=False,
        )
        return df

    market_symbol = _to_market_symbol(symbol)
    df = _attempt(
        "stock_zh_a_daily",
        lambda: _normalize_sina_daily_df(
            ak.stock_zh_a_daily(
                symbol=market_symbol,
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
        ),
    )
    if df is not None:
        _record_provider_use(
            provider="stock_zh_a_daily",
            symbol=market_symbol,
            start=start,
            end=end,
            period="daily",
            adjust=adjust,
            is_fallback=True,
        )
        return df

    df = _attempt(
        "stock_zh_a_hist_tx",
        lambda: _normalize_tx_daily_df(
            ak.stock_zh_a_hist_tx(
                symbol=market_symbol,
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
        ),
    )
    if df is not None:
        _record_provider_use(
            provider="stock_zh_a_hist_tx",
            symbol=market_symbol,
            start=start,
            end=end,
            period="daily",
            adjust=adjust,
            is_fallback=True,
        )
        return df

    raise ValueError("Daily data sources failed: " + " | ".join(errors))


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
    ts = pd.to_datetime(out["ts"], errors="coerce")
    out["ts"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    out.loc[ts.isna(), "ts"] = None

    for c in cols:
        series = pd.to_numeric(out[c], errors="coerce")
        series = series.replace([np.inf, -np.inf], np.nan)
        out[c] = series.apply(lambda x: None if pd.isna(x) else float(x))

    out = out.where(pd.notna(out), None)

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
    period_key = str(period or "daily").lower()
    errors: list[str] = []

    def _attempt(name: str, fn) -> Optional[pd.DataFrame]:
        try:
            return fn()
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            return None

    df = _attempt(
        "stock_zh_a_hist",
        lambda: _normalize_hist_df(
            ak.stock_zh_a_hist(
                symbol=symbol.strip(),
                start_date=start,
                end_date=end,
                period=period_key,
                adjust=adjust,
            )
        ),
    )
    if df is not None:
        _record_provider_use(
            provider="stock_zh_a_hist",
            symbol=symbol,
            start=start,
            end=end,
            period=period_key,
            adjust=adjust,
            is_fallback=False,
        )
        return df

    market_symbol = _to_market_symbol(symbol)

    df = _attempt(
        "stock_zh_a_daily",
        lambda: _normalize_sina_daily_df(
            ak.stock_zh_a_daily(
                symbol=market_symbol,
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
        ),
    )
    if df is not None:
        if period_key != "daily":
            df = _resample_period(df, period_key)
        _record_provider_use(
            provider="stock_zh_a_daily",
            symbol=market_symbol,
            start=start,
            end=end,
            period=period_key,
            adjust=adjust,
            is_fallback=True,
        )
        return df

    df = _attempt(
        "stock_zh_a_hist_tx",
        lambda: _normalize_tx_daily_df(
            ak.stock_zh_a_hist_tx(
                symbol=market_symbol,
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
        ),
    )
    if df is not None:
        if period_key != "daily":
            df = _resample_period(df, period_key)
        _record_provider_use(
            provider="stock_zh_a_hist_tx",
            symbol=market_symbol,
            start=start,
            end=end,
            period=period_key,
            adjust=adjust,
            is_fallback=True,
        )
        return df

    raise ValueError("Period data sources failed: " + " | ".join(errors))


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
