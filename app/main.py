# app/main.py
from __future__ import annotations

from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.config import settings

from app.storage.db import make_engine, make_session_factory, init_db, session_scope
from app.storage.repo import ExecutionRepo

from app.services.market_data import build_market_provider
from app.services.llm_client import OpenAICompatLLM
from app.services.risk import RiskManager
from app.services.broker import PaperBroker
from app.services.strategy import StrategyEngine 
from app.services.trade_history_db import TradeHistoryDB

from app.services.backtest import BacktestParams, run_backtest, build_report_html
from app.services.stock_api import fetch_cn_minute_bars, bars_to_payload

from app.models_llm import LLMChatRequest, LLMChatResponse, ChatMessage
from app.models import TradingContextIn

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
    print(f"DATABASE_URL    = {settings.database_url}")

    # init DB
    await init_db(db_engine)

    yield

    await db_engine.dispose()

    # -------- shutdown --------
    print("🛑 xtrader shutting down...")


# -------------------------
# App
# -------------------------
app = FastAPI(
    title="xtrader",
    lifespan=lifespan,
)

BASE_DIR = Path(__file__).resolve().parents[1]  # /xtrader
STATIC_DIR = BASE_DIR / "static"

if STATIC_DIR.exists() and STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def normalize_cn_symbol(symbol: str) -> str:
    """
    Normalize A-share symbol:
    - keeps only 6-digit code like '600519', '000001', '301xxx'
    - accepts formats like '600519.SH' -> '600519'
    """
    s = symbol.strip().upper()
    if "." in s:
        s = s.split(".", 1)[0]
    if len(s) != 6 or not s.isdigit():
        raise HTTPException(status_code=400, detail=f"Invalid A-share symbol: {symbol} (expect 6-digit code)")
    return s
 
# -------------------------
# Wire core services
# -------------------------
db_engine = make_engine(settings.database_url)
session_factory = make_session_factory(db_engine)

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

# New: keep trade history for LLM self-correction feedback
trade_history = TradeHistoryDB(session_factory, max_records=10)

trading_engine = StrategyEngine(
    market=market,
    llm=llm,
    risk=risk,
    broker=broker,
    fixed_amount_usd=settings.fixed_trade_amount_usd,
    trade_history=trade_history,
    timezone_name="Asia/Shanghai",
    lot_size=100,
    max_order_shares=1000,
    fees_bps_est=5,
    slippage_bps_est=5,
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


# -------------------------
# Updated trading routes (account-aware)
# -------------------------
@app.post("/signal/{symbol}", tags=["trading"])
async def signal(symbol: str, ctx: TradingContextIn):
    """
    NEW (account-aware):
    Build prompt including account_state + execution_constraints + trade_history + bars_5m,
    then request a TradeSignal JSON from LLM.
    """
    code = normalize_cn_symbol(symbol)
    try:
        account = trading_engine.__class__.__mro__  # no-op; keeps linters calm if you later refactor imports
        # Convert to StrategyEngine.AccountState dataclass via prompt_builder.AccountState signature
        # (StrategyEngine expects prompt_builder.AccountState)
        from app.services.prompt_builder import AccountState as AccountStateDC

        acc = AccountStateDC(
            cash_cny=ctx.account_state.cash_cny,
            position_shares=ctx.account_state.position_shares,
            avg_cost_cny=ctx.account_state.avg_cost_cny,
            unrealized_pnl_cny=ctx.account_state.unrealized_pnl_cny,
        )
        sig = await trading_engine.get_signal(code, account=acc, now_ts=ctx.now_ts)
        return sig
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"signal_error: {e}")


@app.post("/execute/{symbol}", tags=["trading"])
async def execute(symbol: str, ctx: TradingContextIn):
    """
    NEW (account-aware) paper execute:
    - If signal is HOLD: returns ok=True, no broker order.
    - Else: if passes RiskManager guardrails -> place paper order.
    Also appends a feedback record to trade_history (best-effort).
    """
    code = normalize_cn_symbol(symbol)
    try:
        from app.services.prompt_builder import AccountState as AccountStateDC

        acc = AccountStateDC(
            cash_cny=ctx.account_state.cash_cny,
            position_shares=ctx.account_state.position_shares,
            avg_cost_cny=ctx.account_state.avg_cost_cny,
            unrealized_pnl_cny=ctx.account_state.unrealized_pnl_cny,
        )
        res = await trading_engine.maybe_execute(code, account=acc, now_ts=ctx.now_ts)

        async with session_scope(session_factory) as s:
            repo = ExecutionRepo(s)
            await repo.add_execution(
                symbol=code,
                ts=res.ts,
                action=res.action,
                notional_usd=res.notional_usd,
                paper=res.paper,
                ok=res.ok,
                message=res.message,
                signal=(res.details.get("signal") if isinstance(res.details, dict) else None),
                snapshot=(res.details.get("snapshot") if isinstance(res.details, dict) else None),
                broker_details=(res.details if isinstance(res.details, dict) else None),
            )
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"execute_error: {e}")


@app.get("/trade_history/{symbol}", tags=["trading"])
async def get_trade_history(symbol: str, limit: int = Query(50, ge=1, le=200)):
    code = normalize_cn_symbol(symbol)
    recs = await trade_history.list_records(code, limit=limit)
    return {"symbol": code, "count": len(recs), "records": recs}

@app.post("/trade_history/{symbol}/clear", tags=["trading"])
async def clear_trade_history(symbol: str):
    code = normalize_cn_symbol(symbol)
    n = await trade_history.clear(code)
    return {"ok": True, "symbol": code, "deleted": n}


# -------------------------
# Stock bars endpoint (unchanged)
# -------------------------
@app.get("/api/stock/bars", tags=["trading"])
async def api_stock_bars(
    symbol: str,
    start: str,
    end: str,
    freq: str = "5",
    limit: int = 2000,
):
    code = normalize_cn_symbol(symbol)
    try:
        df = await run_in_threadpool(
            fetch_cn_minute_bars,
            symbol=code,
            start=start,
            end=end,
            freq=freq,
            limit=limit,
        )
        payload = bars_to_payload(df)
        payload.update({"symbol": code, "freq": freq, "start": start, "end": end})
        return payload
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"stock_bars_error: {e}")


# -------------------------
# Backtest UI + API (unchanged)
# -------------------------
@app.get("/backtest", response_class=HTMLResponse, tags=["backtest"])
async def backtest_page():
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


# -------------------------
# LLM passthrough endpoint (unchanged)
# -------------------------
@app.post("/api/llm/chat", response_model=LLMChatResponse, tags=["llm"])
async def api_llm_chat(req: LLMChatRequest):
    """
    Direct chat endpoint to your configured OpenAI-compatible LLM (DeepSeek/Qwen/etc).

    - If json_schema provided -> structured output in output_json
    - Else -> plain text in output_text
    """
    try:
        system_prompt = req.system or "You are a helpful trading assistant."

        msgs: list[ChatMessage] = [ChatMessage(**m.model_dump()) for m in req.messages]

        if req.user:
            msgs.append(ChatMessage(role="user", content=req.user))

        oa_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        oa_messages += [{"role": m.role, "content": m.content} for m in msgs]

        if req.json_schema:
            latest_user = req.user or (msgs[-1].content if msgs else "")
            obj = await llm.chat_json(system_prompt, latest_user, req.json_schema)

            if hasattr(obj, "model_dump"):
                obj = obj.model_dump()

            return LLMChatResponse(
                ok=True,
                provider=settings.llm_provider,
                model=settings.llm_model,
                output_json=obj,
            )

        text = await llm.chat_messages(
            oa_messages,
            temperature=req.temperature or 0.7,
            max_tokens=req.max_tokens,
        )
        return LLMChatResponse(
            ok=True,
            provider=settings.llm_provider,
            model=settings.llm_model,
            output_text=text,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"llm_chat_error: {e}")
