# app/services/backtest.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import vectorbt as vbt
import akshare as ak


@dataclass
class BacktestParams:
    symbol: str
    freq: Literal["1", "5", "15", "30", "60"] = "5"
    start: str = "2025-12-01 09:30:00"
    end: str = "2025-12-31 15:00:00"

    # sizing
    fixed_notional: float = 200.0     # CNY per trade (approx)
    lot_size: int = 100              # CN-A usually 100 shares

    # controls
    max_trades_per_day: int = 10
    fee_bps: float = 5.0
    slippage_bps: float = 5.0


_ALLOWED_FREQ = {"1", "5", "15", "30", "60"}


def _parse_dt(s: str) -> pd.Timestamp:
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid datetime: {s} (expected 'YYYY-MM-DD HH:MM:SS')")
    return ts


def _detect_time_col(df: pd.DataFrame) -> str:
    for c in ["时间", "日期", "datetime", "date", "time", "ts"]:
        if c in df.columns:
            return c
    raise ValueError(f"Unexpected minute columns (no time col): {list(df.columns)}")


def fetch_minute_bars(params: BacktestParams) -> pd.DataFrame:
    """
    Fetch intraday minute bars via AkShare/Eastmoney.
    Minute-history range may be limited depending on upstream.
    """
    if params.freq not in _ALLOWED_FREQ:
        raise ValueError(f"Invalid freq={params.freq}, allowed: {sorted(_ALLOWED_FREQ)}")

    start_ts = _parse_dt(params.start)
    end_ts = _parse_dt(params.end)
    if end_ts <= start_ts:
        raise ValueError("end must be greater than start")

    df = ak.stock_zh_a_hist_min_em(
        symbol=params.symbol,
        start_date=params.start,
        end_date=params.end,
        period=params.freq,
        adjust="",
    )

    if df is None or df.empty:
        raise ValueError("No minute data returned. Try a closer date range or larger freq (e.g., 15/30/60).")

    tcol = _detect_time_col(df)
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol)

    # de-dup timestamps (can happen)
    df = df.drop_duplicates(subset=[tcol], keep="last").set_index(tcol)

    # normalize columns
    rename_map = {}
    for cn, en in [
        ("开盘", "open"),
        ("最高", "high"),
        ("最低", "low"),
        ("收盘", "close"),
        ("成交量", "volume"),
        ("成交额", "amount"),
        ("均价", "vwap"),
        ("最新价", "close"),
    ]:
        if cn in df.columns:
            rename_map[cn] = en
    df = df.rename(columns=rename_map)

    # numeric coercion
    for col in ["open", "high", "low", "close", "volume", "amount", "vwap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"])
    if df.empty:
        raise ValueError("All close prices are NaN after normalization.")

    return df


def placeholder_signals(
    df: pd.DataFrame,
    *,
    lookback: int = 20,
    vol_window: int = 30,
    vol_quantile: float = 0.4,
) -> Tuple[pd.Series, pd.Series]:
    """
    Mean-reversion placeholder:
    - low-vol regime filter (rolling std quantile)
    - buy when price < MA - k*STD
    - sell when price > MA + k*STD
    """
    close = df["close"].astype(float).copy()
    ret = close.pct_change().fillna(0.0)

    vol = ret.rolling(vol_window, min_periods=max(2, vol_window // 3)).std()
    vol_th = float(vol.quantile(vol_quantile)) if not vol.dropna().empty else float("inf")
    low_vol = vol <= vol_th

    ma = close.rolling(lookback, min_periods=max(2, lookback // 3)).mean()
    sd = close.rolling(lookback, min_periods=max(2, lookback // 3)).std().replace(0, np.nan)

    k = 1.0
    buy = (low_vol & (close < (ma - k * sd))).fillna(False)
    sell = (low_vol & (close > (ma + k * sd))).fillna(False)

    # avoid simultaneous entry/exit on same bar
    both = buy & sell
    if both.any():
        sell = sell & (~both)

    return buy.astype(bool), sell.astype(bool)


def enforce_trade_limit_per_day(
    index: pd.DatetimeIndex,
    entries: pd.Series,
    exits: pd.Series,
    max_trades_per_day: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    Simple per-day trade cap: keep only first N entry signals each day.
    Exits are kept (naively) to allow closing positions.
    """
    max_trades_per_day = max(0, int(max_trades_per_day))

    ent = pd.Series(False, index=index)
    ex = pd.Series(False, index=index)

    df = pd.DataFrame({"entries": entries.astype(bool), "exits": exits.astype(bool)}, index=index)
    df["day"] = df.index.date

    for _, g in df.groupby("day", sort=True):
        ent_idx = g.index[g["entries"]].tolist()
        ent_keep = ent_idx[:max_trades_per_day] if max_trades_per_day > 0 else []
        ent.loc[ent_keep] = True
        ex.loc[g.index[g["exits"]]] = True

    return ent, ex


def _build_size_series(
    close: pd.Series,
    *,
    fixed_notional: float,
    lot_size: int,
) -> pd.Series:
    """
    Convert fixed_notional (CNY) into CN-A share size per bar, rounded down to lot_size.
    """
    lot_size = max(1, int(lot_size))
    fixed_notional = float(fixed_notional)

    px = close.astype(float)
    raw_shares = np.floor((fixed_notional / px).replace([np.inf, -np.inf], np.nan)).fillna(0.0)
    lots = np.floor(raw_shares / lot_size)
    shares = (lots * lot_size).astype(int)

    # if notional too small, shares become 0; vectorbt treats 0 as "no order"
    return shares


def run_backtest(params: BacktestParams) -> Dict[str, Any]:
    bars = fetch_minute_bars(params)
    close = bars["close"].astype(float)

    entries, exits = placeholder_signals(bars)
    entries2, exits2 = enforce_trade_limit_per_day(close.index, entries, exits, params.max_trades_per_day)

    # sizing
    size = _build_size_series(close, fixed_notional=params.fixed_notional, lot_size=params.lot_size)

    # costs
    fee = float(params.fee_bps) / 1e4
    slippage = float(params.slippage_bps) / 1e4

    # Portfolio
    pf = vbt.Portfolio.from_signals(
        close,
        entries=entries2,
        exits=exits2,
        size=size,               # shares per trade
        size_type="amount",      # "amount" = number of shares/contracts
        fees=fee,
        slippage=slippage,
        init_cash=10_000.0,
        freq=f"{params.freq}min",
    )

    stats = pf.stats().to_dict()
    equity = pf.value().to_frame(name="equity")
    drawdown = pf.drawdown().to_frame(name="drawdown")

    return {
        "symbol": params.symbol,
        "freq": params.freq,
        "start": params.start,
        "end": params.end,
        "fixed_notional": float(params.fixed_notional),
        "lot_size": int(params.lot_size),
        "fee_bps": float(params.fee_bps),
        "slippage_bps": float(params.slippage_bps),
        "stats": stats,
        "equity": equity.reset_index().rename(columns={"index": "ts"}).to_dict(orient="records"),
        "drawdown": drawdown.reset_index().rename(columns={"index": "ts"}).to_dict(orient="records"),
        "note": "Signals are placeholder; replace with your LLM-driven signals later.",
    }


def build_report_html(result: Dict[str, Any]) -> str:
    eq = pd.DataFrame(result.get("equity", []))
    dd = pd.DataFrame(result.get("drawdown", []))

    if not eq.empty:
        eq["ts"] = pd.to_datetime(eq["ts"], errors="coerce")
    if not dd.empty:
        dd["ts"] = pd.to_datetime(dd["ts"], errors="coerce")

    fig1 = go.Figure()
    if not eq.empty:
        fig1.add_trace(go.Scatter(x=eq["ts"], y=eq["equity"], mode="lines", name="Equity"))

    fig2 = go.Figure()
    if not dd.empty:
        fig2.add_trace(go.Scatter(x=dd["ts"], y=dd["drawdown"], mode="lines", name="Drawdown"))

    stats = result.get("stats", {}) or {}
    items = list(stats.items())[:80]
    stats_rows = "".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in items])

    symbol = result.get("symbol", "")
    freq = result.get("freq", "")
    start = result.get("start", "")
    end = result.get("end", "")
    note = result.get("note", "")

    fixed_notional = result.get("fixed_notional", "")
    lot_size = result.get("lot_size", "")
    fee_bps = result.get("fee_bps", "")
    slippage_bps = result.get("slippage_bps", "")

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Backtest Report - {symbol}</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td {{ border-bottom: 1px solid #eee; padding: 6px; font-size: 12px; }}
    .muted {{ color: #666; font-size: 12px; margin-bottom: 8px; }}
  </style>
</head>
<body>
  <h2>Backtest Report - {symbol}</h2>
  <div class="muted">freq={freq} | {start} → {end}</div>
  <div class="muted">fixed_notional={fixed_notional} | lot_size={lot_size} | fee_bps={fee_bps} | slippage_bps={slippage_bps}</div>
  <div class="muted">{note}</div>

  <div class="grid">
    <div class="card">
      <h3>Equity Curve</h3>
      <div id="equity"></div>
    </div>

    <div class="card">
      <h3>Drawdown</h3>
      <div id="drawdown"></div>
    </div>

    <div class="card">
      <h3>Key Stats</h3>
      <table>{stats_rows}</table>
    </div>
  </div>

  <script>
    const fig1 = {fig1.to_json()};
    const fig2 = {fig2.to_json()};
    Plotly.newPlot('equity', fig1.data, fig1.layout, {{responsive:true}});
    Plotly.newPlot('drawdown', fig2.data, fig2.layout, {{responsive:true}});
  </script>
</body>
</html>
"""
