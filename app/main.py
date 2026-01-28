# app/main.py
from __future__ import annotations

from pathlib import Path
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.logging_config import logger

from app.services.cache import build_cache

from app.storage.db import make_engine, make_session_factory, init_db, session_scope

from app.services.market_data import build_market_provider
from app.services.llm_client import OpenAICompatLLM
from app.services.risk import RiskManager
from app.services.broker import PaperBroker
from app.services.strategy import StrategyEngine
from app.services.trade_history_db import TradeHistoryDB
from app.services.backtest import BacktestParams, run_backtest, build_report_html
from app.services.stock_api import fetch_cn_minute_bars, bars_to_payload
from app.services.auth import (
    AuthenticatedUser,
    clear_session_cookie,
    get_auth_service,
    set_session_cookie,
    get_session_username,
    hash_password,
    is_superuser,
)

from app.models_llm import LLMChatRequest, LLMChatResponse, ChatMessage
from app.models import TradingContextIn, AccountState
from app.models_auth import AuthStatus, LoginRequest, SignupRequest, UserPublic
from app.models_admin import AdminUserCreate, AdminUserUpdate, AdminUserList, ApiUsageSummary
from app.storage.auth_repo import UserRepo
from app.storage.usage_repo import ApiUsageRepo

from app.utils.textutils import normalize_cn_symbol

# -------------------------
# Auth helpers
# -------------------------
async def require_user(auth=Depends(get_auth_service)) -> AuthenticatedUser:
    return await auth.current_user()

async def require_superuser(user: AuthenticatedUser = Depends(require_user)) -> AuthenticatedUser:
    if not is_superuser(user):
        raise HTTPException(status_code=403, detail="superuser required")
    return user

def _to_user_public(user) -> UserPublic:
    return UserPublic(
        id=int(user.id),
        username=str(user.username),
        created_at=user.created_at.isoformat() if user.created_at else None,
        last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
    )

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

    # 1) DB engine + session factory
    db_engine = make_engine(settings.database_url)
    session_factory = make_session_factory(db_engine)

    # 2) Init DB schema + SQLite pragmas
    await init_db(db_engine)

    # build cache once (single worker)
    cache = build_cache(backend=settings.cache_backend, redis_url=settings.cache_redis_url)
    app.state.cache = cache

    # 3) Save to app.state
    app.state.db_engine = db_engine
    app.state.session_factory = session_factory

    # 4) DB-backed trade history
    trade_history_db = TradeHistoryDB(session_factory, max_records=10)
    app.state.trade_history_db = trade_history_db

    # 5) Core services
    market = build_market_provider(
        settings.market_provider,          # whatever you use, e.g. "akshare"
        cache=cache,             # Memory now, Redis later
        # spot_ttl_sec=settings.cache_spot_ttl_sec,
        orderbook_ttl_sec=settings.cache_orderbook_ttl_sec,
        bars_ttl_sec=settings.cache_bars_ttl_sec,
    )

    llm = OpenAICompatLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
    app.state.llm = llm  # ✅ used by /api/llm/chat

    risk = RiskManager(
        max_trades_per_day=settings.max_trades_per_day,
        cooldown_seconds=settings.trade_cooldown_seconds,
        min_confidence=settings.min_confidence,
        max_position_value_cny=settings.max_position_value_cny,
    )
    app.state.risk = risk

    broker = PaperBroker()
    app.state.broker = broker

    # 6) Trading engine (uses DB-backed trade history)
    trading_engine = StrategyEngine(
        market=market,
        llm=llm,
        risk=risk,
        broker=broker,
        fixed_lots=settings.fixed_slots,     # keep your existing setting name
        trade_history_db=trade_history_db,
        trade_history_limit=10,
    )
    app.state.trading_engine = trading_engine

    try:
        yield
    finally:
        # -------- shutdown --------
        print("🛑 xtrader shutting down...")
        await cache.close()
        await db_engine.dispose()


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

# -------------------------
# API usage tracking
# -------------------------
@app.middleware("http")
async def api_usage_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000.0

    path = request.url.path
    if path.startswith("/static") or path == "/favicon.ico":
        return response

    try:
        session_factory = request.app.state.session_factory
        token = request.cookies.get(settings.auth_cookie_name)
        username = get_session_username(token) if token else None
        async with session_scope(session_factory) as s:
            repo = ApiUsageRepo(s)
            await repo.add(
                path=path,
                method=request.method,
                status_code=response.status_code,
                username=username,
                duration_ms=duration_ms,
            )
    except Exception:
        logger.exception("api_usage_log_error | path=%s", path)

    return response

# -------------------------
# Basic endpoints
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = STATIC_DIR / "home.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="home.html not found in /static")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.get("/health", tags=["system"])
async def health():
    return {"ok": True, "app": "xtrader"}


# -------------------------
# Public auth pages
# -------------------------
@app.get("/signup", response_class=HTMLResponse, tags=["auth-ui"])
async def signup_page():
    html_path = STATIC_DIR / "signup.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="signup.html not found in /static")
    return html_path.read_text(encoding="utf-8")


@app.get("/login", response_class=HTMLResponse, tags=["auth-ui"])
async def login_page():
    html_path = STATIC_DIR / "login.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="login.html not found in /static")
    return html_path.read_text(encoding="utf-8")


# -------------------------
# Auth routes
# -------------------------
@app.post("/auth/signup", response_model=AuthStatus, tags=["auth"])
async def auth_signup(req: SignupRequest, request: Request):
    auth = get_auth_service(request)
    user = await auth.signup(username=req.username, password=req.password)
    return AuthStatus(ok=True, user=user.to_public())


@app.post("/auth/login", response_model=AuthStatus, tags=["auth"])
async def auth_login(req: LoginRequest, request: Request, response: Response):
    auth = get_auth_service(request)
    user, token = await auth.login(username=req.username, password=req.password)
    set_session_cookie(response, token)
    return AuthStatus(ok=True, user=user.to_public())


@app.post("/auth/logout", response_model=AuthStatus, tags=["auth"])
async def auth_logout(response: Response):
    clear_session_cookie(response)
    return AuthStatus(ok=True, detail="logged out")


@app.get("/auth/me", response_model=AuthStatus, tags=["auth"])
async def auth_me(user: AuthenticatedUser = Depends(require_user)):
    return AuthStatus(ok=True, user=user.to_public())


# -------------------------
# Admin UI + APIs (superuser only)
# -------------------------
@app.get("/admin", response_class=HTMLResponse, tags=["admin-ui"])
async def admin_page(user: AuthenticatedUser = Depends(require_superuser)):
    html_path = STATIC_DIR / "admin_users.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="admin_users.html not found in /static")
    return html_path.read_text(encoding="utf-8")


@app.get("/admin/users-ui", response_class=HTMLResponse, tags=["admin-ui"])
async def admin_users_page(user: AuthenticatedUser = Depends(require_superuser)):
    html_path = STATIC_DIR / "admin_users.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="admin_users.html not found in /static")
    return html_path.read_text(encoding="utf-8")


@app.get("/admin/usage-ui", response_class=HTMLResponse, tags=["admin-ui"])
async def admin_usage_page(user: AuthenticatedUser = Depends(require_superuser)):
    html_path = STATIC_DIR / "admin_usage.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="admin_usage.html not found in /static")
    return html_path.read_text(encoding="utf-8")


@app.get("/admin/users", response_model=AdminUserList, tags=["admin"])
async def admin_list_users(
    q: Optional[str] = Query(None, description="Search by username"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: AuthenticatedUser = Depends(require_superuser),
):
    async with session_scope(app.state.session_factory) as s:
        repo = UserRepo(s)
        items = await repo.list_users(limit=limit, offset=offset, search=q)
        total = await repo.count_users(search=q)
        return AdminUserList(
            items=[_to_user_public(u) for u in items],
            total=total,
            limit=limit,
            offset=offset,
        )


@app.post("/admin/users", response_model=UserPublic, tags=["admin"])
async def admin_create_user(
    req: AdminUserCreate,
    user: AuthenticatedUser = Depends(require_superuser),
):
    pwd_hash, salt = hash_password(req.password)
    async with session_scope(app.state.session_factory) as s:
        repo = UserRepo(s)
        existing = await repo.get_by_username(req.username)
        if existing:
            raise HTTPException(status_code=409, detail="username already exists")
        new_user = await repo.create_user(username=req.username, password_hash=pwd_hash, salt=salt)
        return _to_user_public(new_user)


@app.patch("/admin/users/{user_id}", response_model=UserPublic, tags=["admin"])
async def admin_update_user(
    user_id: int,
    req: AdminUserUpdate,
    user: AuthenticatedUser = Depends(require_superuser),
):
    async with session_scope(app.state.session_factory) as s:
        repo = UserRepo(s)
        target = await repo.get_by_id(user_id)
        if not target:
            raise HTTPException(status_code=404, detail="user not found")

        if req.username and req.username != target.username:
            existing = await repo.get_by_username(req.username)
            if existing:
                raise HTTPException(status_code=409, detail="username already exists")
            target.username = req.username

        if req.password:
            pwd_hash, salt = hash_password(req.password)
            target.password_hash = pwd_hash
            target.salt = salt

        s.add(target)
        return _to_user_public(target)


@app.delete("/admin/users/{user_id}", tags=["admin"])
async def admin_delete_user(
    user_id: int,
    user: AuthenticatedUser = Depends(require_superuser),
):
    if int(user_id) == int(user.id):
        raise HTTPException(status_code=400, detail="cannot delete the current superuser")
    async with session_scope(app.state.session_factory) as s:
        repo = UserRepo(s)
        deleted = await repo.delete_user(user_id)
        if deleted <= 0:
            raise HTTPException(status_code=404, detail="user not found")
        return {"ok": True, "deleted": deleted}


@app.get("/admin/api-usage", response_model=ApiUsageSummary, tags=["admin"])
async def admin_api_usage(
    days: int = Query(7, ge=1, le=90),
    user: AuthenticatedUser = Depends(require_superuser),
):
    async with session_scope(app.state.session_factory) as s:
        repo = ApiUsageRepo(s)
        series = await repo.list_daily_counts(days=days)
        top_paths = await repo.top_paths(days=days, limit=8)
        top_users = await repo.top_users(days=days, limit=8)
        return ApiUsageSummary(days=days, series=series, top_paths=top_paths, top_users=top_users)


@app.get("/risk", tags=["trading"])
async def risk_status(user: AuthenticatedUser = Depends(require_user)):
    risk = app.state.risk
    return {
        "max_trades_per_day": settings.max_trades_per_day,
        "trades_left_today": risk.trades_left(),
        "cooldown_seconds": settings.trade_cooldown_seconds,
        "min_confidence": settings.min_confidence,
        "fixed_trade_amount_cny": settings.fixed_trade_amount_cny,
    }


# -------------------------
# Trading routes (account-aware)
# -------------------------
@app.get("/signal-ui", response_class=HTMLResponse, tags=["ui"])
async def signal_ui_page(user: AuthenticatedUser = Depends(require_user)):
    html_path = STATIC_DIR / "signal.html"
    if not html_path.exists():
        raise HTTPException(
            status_code=404,
            detail="signal.html not found. Create /static/signal.html at repo root.",
        )
    return html_path.read_text(encoding="utf-8")

@app.post("/signal/{symbol}", tags=["trading"])
async def signal(
    symbol: str,
    ctx: TradingContextIn,
    user: AuthenticatedUser = Depends(require_user),
):
    """
    Account-aware signal:
    - Body includes account_state + now_ts + optional options overrides
    """
    try:
        code = normalize_cn_symbol(symbol)

        acc = AccountState(
            user_id=int(user.id),
            username=str(user.username),
            account_id=str(ctx.account_state.account_id),
            cash_cny=ctx.account_state.cash_cny,
            position_shares=ctx.account_state.position_shares,
            avg_cost_cny=ctx.account_state.avg_cost_cny,
            unrealized_pnl_cny=ctx.account_state.unrealized_pnl_cny,
        )

        trading_engine: StrategyEngine = app.state.trading_engine

        opt = ctx.options
        sig = await trading_engine.get_signal(
            code,
            account=acc,
            now_ts=ctx.now_ts,
            timezone_name=opt.timezone_name,
            lot_size=int(opt.lot_size),
            max_order_shares=opt.normalized_max_order_shares(),
            fees_bps_est=int(opt.fees_bps_est),
            slippage_bps_est=int(opt.slippage_bps_est),
            market_kwargs=opt.market.to_kwargs(),
            strategy_mode=opt.strategy_mode,
        )
        return sig

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("signal_error | symbol=%s", symbol)
        raise HTTPException(status_code=500, detail=f"signal_error: {type(e).__name__}: {e}")

@app.get("/ui/execute", response_class=HTMLResponse, tags=["ui"])
async def ui_execute():
    html_path = STATIC_DIR / "execute.html"
    if not html_path.exists():
        raise HTTPException(
            status_code=404,
            detail="execute.html not found. Create /static/execute.html at repo root.",
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.post("/execute/{symbol}", tags=["trading"])
async def execute(
    symbol: str,
    ctx: TradingContextIn,
    user: AuthenticatedUser = Depends(require_user),
):
    """
    Account-aware execute (paper for now):
    - HOLD => ok=True, no broker call
    - BUY/SELL => risk-checked, place paper order
    - persists intraday records into DB
    """
    try:
        code = normalize_cn_symbol(symbol)

        acc = AccountState(
            user_id=int(user.id),
            username=str(user.username),
            account_id=str(ctx.account_state.account_id),
            cash_cny=ctx.account_state.cash_cny,
            position_shares=ctx.account_state.position_shares,
            avg_cost_cny=ctx.account_state.avg_cost_cny,
            unrealized_pnl_cny=ctx.account_state.unrealized_pnl_cny,
        )

        trading_engine: StrategyEngine = app.state.trading_engine

        opt = ctx.options
        res = await trading_engine.maybe_execute(
            code,
            account=acc,
            now_ts=ctx.now_ts,
            timezone_name=opt.timezone_name,
            lot_size=int(opt.lot_size),
            max_order_shares=opt.normalized_max_order_shares(),
            fees_bps_est=int(opt.fees_bps_est),
            slippage_bps_est=int(opt.slippage_bps_est),
            market_kwargs=opt.market.to_kwargs(),
            strategy_mode=opt.strategy_mode,
        )
        return res

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"execute_error: {e}")


@app.get("/trade_history/{symbol}", tags=["trading"])
async def get_trade_history(
    symbol: str,
    now_ts: str = Query(..., description="YYYY-MM-DD HH:MM:SS"),
    limit: int = Query(50, ge=1, le=200),
    include_json: bool = Query(False),
    account_id: str = Query("default", min_length=1, max_length=64, description="Account identifier"),
    user: AuthenticatedUser = Depends(require_user),
):
    code = normalize_cn_symbol(symbol)
    trade_history_db: TradeHistoryDB = app.state.trade_history_db
    recs = await trade_history_db.list_intraday_today(
        code,
        now_ts=now_ts,
        user_id=int(user.id),
        account_id=str(account_id),
        limit=limit,
        include_json=include_json,
    )
    return {"symbol": code, "account_id": account_id, "count": len(recs), "records": recs}


@app.post("/trade_history/{symbol}/clear", tags=["trading"])
async def clear_trade_history(
    symbol: str,
    account_id: str = Query("default", min_length=1, max_length=64, description="Account identifier"),
    user: AuthenticatedUser = Depends(require_user),
):
    code = normalize_cn_symbol(symbol)
    trade_history_db: TradeHistoryDB = app.state.trade_history_db
    n = await trade_history_db.clear(code, user_id=int(user.id), account_id=str(account_id))
    return {"ok": True, "symbol": code, "account_id": account_id, "deleted": n}


# TODO: 增加隔日交易，隔周交易，和隔月交易的接口

# -------------------------
# Stock bars endpoint
# -------------------------
@app.get("/api/stock/bars", tags=["trading"])
async def api_stock_bars(
    symbol: str,
    start: str,
    end: str,
    freq: str = "1",
    limit: Optional[int] = None, 
    user: AuthenticatedUser = Depends(require_user),
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
        payload.update({
            "symbol": code,
            "freq": freq,
            "start": start,
            "end": end,
            "limit": limit,
            "count": len(df),
        })
        return payload
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"stock_bars_error: {e}")


# -------------------------
# Backtest UI + API
# -------------------------
@app.get("/backtest", response_class=HTMLResponse, tags=["backtest"])
async def backtest_page(user: AuthenticatedUser = Depends(require_user)):
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
    user: AuthenticatedUser = Depends(require_user),
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
    user: AuthenticatedUser = Depends(require_user),
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
# LLM passthrough endpoint
# -------------------------
@app.post("/api/llm/chat", response_model=LLMChatResponse, tags=["llm"])
async def api_llm_chat(req: LLMChatRequest, user: AuthenticatedUser = Depends(require_user)):
    """
    Direct chat endpoint to your configured OpenAI-compatible LLM (DeepSeek/Qwen/etc).

    - If json_schema provided -> structured output in output_json
    - Else -> plain text in output_text
    """
    try:
        llm: OpenAICompatLLM = app.state.llm

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
