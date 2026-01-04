# app/main.py
from __future__ import annotations

from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.services.market_data import build_market_provider
from app.services.llm_client import OpenAICompatLLM
from app.services.risk import RiskManager
from app.services.broker import PaperBroker
from app.services.strategy import StrategyEngine
from app.services.backtest import BacktestParams, run_backtest, build_report_html

# -------------------------
# Lifespan (startup / shutdown)
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # -------- startup --------
    print("🚀 xtrader starting up...")
    print(f"MARKET_PROVIDER = {settings.market_provider}")
    print(f"LLM_BASE_URL    = {settings.llm_base_url}")
    print(f"LLM_MODEL       = {settings.llm_model}")

    # You can also initialize shared resources here
    # e.g. DB connections, async clients, caches

    yield

    # -------- shutdown --------
    print("🛑 xtrader shutting down...")
    # Close DB pools, async clients, etc.


# -------------------------
# App
# -------------------------
app = FastAPI(
    title="xtrader",
    lifespan=lifespan,
) 

BASE_DIR = Path(__file__).resolve().parents[1]  # /xtrader
STATIC_DIR = BASE_DIR / "static"

# Mount /static if it exists (recommended at repo root)
if STATIC_DIR.exists() and STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    # Don't crash; you can still run API-only mode
    # Create /static/backtest.html later
    pass


def normalize_cn_symbol(symbol: str) -> str:
    """
    Normalize A-share symbol:
    - keeps only 6-digit code like '600519', '000001', '301xxx'
    - you can extend this to accept '600519.SH' -> '600519' later
    """
    s = symbol.strip().upper()
    # Accept formats like "600519.SH" / "000001.SZ"
    if "." in s:
        s = s.split(".", 1)[0]
    if len(s) != 6 or not s.isdigit():
        raise HTTPException(status_code=400, detail=f"Invalid A-share symbol: {symbol} (expect 6-digit code)")
    return s


# -------------------------
# Wire core services
# -------------------------
market = build_market_provider(settings.market_provider)

llm = OpenAICompatLLM(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    model=settings.llm_model,
)

risk = RiskManager(
    max_trades_per_day=settings.max_trades_per_day,
    cooldown_seconds=settings.trade_cooldown_seconds,
    min_confidence=settings.min_confidence,
)

broker = PaperBroker()

engine = StrategyEngine(
    market=market,
    llm=llm,
    risk=risk,
    broker=broker,
    fixed_amount_usd=settings.fixed_trade_amount_usd,
)


# -------------------------
# Basic endpoints
# ------------------------- 

@app.get("/health", tags=["system"])
async def health():
    return {"ok": True, "app": "xtrader"}


@app.get("/risk", tags=["trading"])
async def risk_status():
    return {
        "max_trades_per_day": settings.max_trades_per_day,
        "trades_left_today": risk.trades_left(),
        "cooldown_seconds": settings.trade_cooldown_seconds,
        "min_confidence": settings.min_confidence,
        "fixed_trade_amount_usd": settings.fixed_trade_amount_usd,
    }


@app.get("/signal/{symbol}", tags=["trading"])
async def signal(symbol: str):
    code = normalize_cn_symbol(symbol)
    try:
        sig = await engine.get_signal(code)
        return sig
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"signal_error: {e}")


@app.post("/execute/{symbol}", tags=["trading"])
async def execute(symbol: str):
    """
    Paper execute if signal passes guardrails.
    Keep paper trading until you add broker auth + additional risk checks.
    """
    code = normalize_cn_symbol(symbol)
    try:
        res = await engine.maybe_execute(code)
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"execute_error: {e}")


# -------------------------
# Backtest UI + API
# -------------------------
@app.get("/backtest", response_class=HTMLResponse, tags=["backtest"])
async def backtest_page():
    """
    Serves a simple HTML page (static/backtest.html).
    """
    html_path = STATIC_DIR / "backtest.html"
    if not html_path.exists():
        raise HTTPException(
            status_code=404,
            detail="backtest.html not found. Create /static/backtest.html at repo root.",
        )
    return html_path.read_text(encoding="utf-8")


@app.get("/api/backtest/run", tags=["backtest"])
async def api_backtest_run(
    symbol: str = Query(..., description="6-digit A-share code, e.g. 600519"),
    start: str = Query(..., description="YYYY-MM-DD HH:MM:SS"),
    end: str = Query(..., description="YYYY-MM-DD HH:MM:SS"),
    freq: str = Query("5", description="Minute frequency: 1/5/15/30/60"),
    fixed_notional: float = Query(200.0, ge=0.0),
    max_trades_per_day: int = Query(10, ge=0, le=100),
):
    code = normalize_cn_symbol(symbol)

    params = BacktestParams(
        symbol=code,
        start=start,
        end=end,
        freq=freq,
        fixed_notional=fixed_notional,
        max_trades_per_day=max_trades_per_day,
    )

    try:
        # run_backtest is sync + can be heavy (AkShare IO + vectorbt)
        result = await run_in_threadpool(run_backtest, params)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"backtest_run_error: {e}")


@app.get("/api/backtest/report", response_class=HTMLResponse, tags=["backtest"])
async def api_backtest_report(
    symbol: str = Query(..., description="6-digit A-share code, e.g. 600519"),
    start: str = Query(..., description="YYYY-MM-DD HH:MM:SS"),
    end: str = Query(..., description="YYYY-MM-DD HH:MM:SS"),
    freq: str = Query("5", description="Minute frequency: 1/5/15/30/60"),
    fixed_notional: float = Query(200.0, ge=0.0),
    max_trades_per_day: int = Query(10, ge=0, le=100),
):
    code = normalize_cn_symbol(symbol)

    params = BacktestParams(
        symbol=code,
        start=start,
        end=end,
        freq=freq,
        fixed_notional=fixed_notional,
        max_trades_per_day=max_trades_per_day,
    )

    try:
        result = await run_in_threadpool(run_backtest, params)
        html = build_report_html(result)
        return html
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"backtest_report_error: {e}")
