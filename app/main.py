# app/main.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.exc import IntegrityError

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
from app.services.stock_api import fetch_cn_minute_bars, fetch_cn_daily_bars, bars_to_payload
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
from app.models_agent import StockStrategyRequest, StockStrategyAdvice, StockStrategyLLM
from app.models_account import (
    TradeAccountCreate,
    TradeAccountUpdate,
    TradeAccountOut,
    TradeAccountDetail,
    TradeAccountList,
    TradePositionCreate,
    TradePositionUpdate,
    TradePositionOut,
)
from app.storage.auth_repo import UserRepo
from app.storage.usage_repo import ApiUsageRepo
from app.storage.account_repo import TradeAccountRepo

from app.utils.textutils import normalize_cn_symbol
from app.utils.timeutils import now_shanghai, fmt_shanghai

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
    trade_history_db = TradeHistoryDB(
        session_factory,
        max_records=int(settings.trade_history_max_records),
    )
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
        trade_history_limit=int(settings.trade_history_prompt_limit),
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


# -------------------------
# Profile UI (account management)
# -------------------------
@app.get("/profile", response_class=HTMLResponse, tags=["profile-ui"])
async def profile_page(user: AuthenticatedUser = Depends(require_user)):
    html_path = STATIC_DIR / "profile.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="profile.html not found in /static")
    return html_path.read_text(encoding="utf-8")


@app.get("/agent/analyst", response_class=HTMLResponse, tags=["agent-ui"])
async def analyst_page(user: AuthenticatedUser = Depends(require_user)):
    html_path = STATIC_DIR / "analyst.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="analyst.html not found in /static")
    return html_path.read_text(encoding="utf-8")


@app.get("/admin/users", response_model=AdminUserList, tags=["admin"])
async def admin_list_users(
    q: Optional[str] = Query(None, description="Search by username"),
    limit: int = Query(min(50, int(settings.trade_history_max_records)), ge=1, le=settings.trade_history_max_records),
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


# -------------------------
# Account management APIs
# -------------------------
def _fmt_dt(dt) -> Optional[str]:
    return dt.isoformat() if dt else None

def _account_out(acc) -> TradeAccountOut:
    return TradeAccountOut(
        account_id=str(acc.account_id),
        name=acc.name,
        cash_cny=float(acc.cash_cny or 0.0),
        base_currency=str(acc.base_currency or "CNY"),
        created_at=_fmt_dt(acc.created_at),
        updated_at=_fmt_dt(acc.updated_at),
    )

def _position_out(pos) -> TradePositionOut:
    return TradePositionOut(
        symbol=str(pos.symbol),
        shares=int(pos.shares or 0),
        avg_cost_cny=float(pos.avg_cost_cny) if pos.avg_cost_cny is not None else None,
        unrealized_pnl_cny=float(pos.unrealized_pnl_cny) if pos.unrealized_pnl_cny is not None else None,
        created_at=_fmt_dt(pos.created_at),
        updated_at=_fmt_dt(pos.updated_at),
    )


def _strategy_schema_hint() -> dict:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "confidence_breakdown": {"type": "object"},
            "action_bias": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
            "plan": {"type": "string"},
            "key_levels": {"type": "array", "items": {"type": "string"}},
            "risks": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
        "required": ["summary", "confidence_breakdown", "action_bias", "plan", "key_levels", "risks", "confidence"],
    }

@app.get("/api/accounts", response_model=TradeAccountList, tags=["accounts"])
async def list_accounts(user: AuthenticatedUser = Depends(require_user)):
    async with session_scope(app.state.session_factory) as s:
        repo = TradeAccountRepo(s)
        items = await repo.list_accounts(user_id=int(user.id))
        return TradeAccountList(items=[_account_out(a) for a in items], total=len(items))

@app.post("/api/accounts", response_model=TradeAccountOut, tags=["accounts"])
async def create_account(req: TradeAccountCreate, user: AuthenticatedUser = Depends(require_user)):
    async with session_scope(app.state.session_factory) as s:
        repo = TradeAccountRepo(s)
        existing = await repo.get_account(user_id=int(user.id), account_id=req.account_id)
        if existing:
            raise HTTPException(status_code=409, detail="account_id already exists")
        try:
            acc = await repo.create_account(
                user_id=int(user.id),
                account_id=req.account_id,
                name=req.name,
                cash_cny=req.cash_cny,
                base_currency=req.base_currency,
            )
            for pos in req.positions or []:
                await repo.create_position(
                    account_pk=acc.id,
                    user_id=int(user.id),
                    symbol=pos.symbol,
                    shares=pos.shares,
                    avg_cost_cny=pos.avg_cost_cny,
                    unrealized_pnl_cny=pos.unrealized_pnl_cny,
                )
        except IntegrityError:
            raise HTTPException(status_code=409, detail="account_id already exists")
        return _account_out(acc)

@app.get("/api/accounts/{account_id}", response_model=TradeAccountDetail, tags=["accounts"])
async def get_account(account_id: str, user: AuthenticatedUser = Depends(require_user)):
    async with session_scope(app.state.session_factory) as s:
        repo = TradeAccountRepo(s)
        acc = await repo.get_account(user_id=int(user.id), account_id=str(account_id))
        if not acc:
            raise HTTPException(status_code=404, detail="account not found")
        positions = await repo.list_positions(account_pk=acc.id, user_id=int(user.id))
        return TradeAccountDetail(
            **_account_out(acc).model_dump(),
            positions=[_position_out(p) for p in positions],
        )

@app.patch("/api/accounts/{account_id}", response_model=TradeAccountOut, tags=["accounts"])
async def update_account(
    account_id: str,
    req: TradeAccountUpdate,
    user: AuthenticatedUser = Depends(require_user),
):
    async with session_scope(app.state.session_factory) as s:
        repo = TradeAccountRepo(s)
        acc = await repo.get_account(user_id=int(user.id), account_id=str(account_id))
        if not acc:
            raise HTTPException(status_code=404, detail="account not found")
        acc = await repo.update_account(
            acc,
            name=req.name,
            cash_cny=req.cash_cny,
            base_currency=req.base_currency,
        )
        return _account_out(acc)

@app.delete("/api/accounts/{account_id}", tags=["accounts"])
async def delete_account(account_id: str, user: AuthenticatedUser = Depends(require_user)):
    async with session_scope(app.state.session_factory) as s:
        repo = TradeAccountRepo(s)
        acc = await repo.get_account(user_id=int(user.id), account_id=str(account_id))
        if not acc:
            raise HTTPException(status_code=404, detail="account not found")
        deleted = await repo.delete_account(account_pk=acc.id, user_id=int(user.id))
        if deleted <= 0:
            raise HTTPException(status_code=404, detail="account not found")
        return {"ok": True, "deleted": deleted, "account_id": account_id}

@app.get("/api/accounts/{account_id}/positions", response_model=list[TradePositionOut], tags=["accounts"])
async def list_positions(account_id: str, user: AuthenticatedUser = Depends(require_user)):
    async with session_scope(app.state.session_factory) as s:
        repo = TradeAccountRepo(s)
        acc = await repo.get_account(user_id=int(user.id), account_id=str(account_id))
        if not acc:
            raise HTTPException(status_code=404, detail="account not found")
        positions = await repo.list_positions(account_pk=acc.id, user_id=int(user.id))
        return [_position_out(p) for p in positions]

@app.post("/api/accounts/{account_id}/positions", response_model=TradePositionOut, tags=["accounts"])
async def create_position(
    account_id: str,
    req: TradePositionCreate,
    user: AuthenticatedUser = Depends(require_user),
):
    async with session_scope(app.state.session_factory) as s:
        repo = TradeAccountRepo(s)
        acc = await repo.get_account(user_id=int(user.id), account_id=str(account_id))
        if not acc:
            raise HTTPException(status_code=404, detail="account not found")
        existing = await repo.get_position(account_pk=acc.id, user_id=int(user.id), symbol=req.symbol)
        if existing:
            raise HTTPException(status_code=409, detail="position already exists")
        try:
            pos = await repo.create_position(
                account_pk=acc.id,
                user_id=int(user.id),
                symbol=req.symbol,
                shares=req.shares,
                avg_cost_cny=req.avg_cost_cny,
                unrealized_pnl_cny=req.unrealized_pnl_cny,
            )
        except IntegrityError:
            raise HTTPException(status_code=409, detail="position already exists")
        return _position_out(pos)

@app.patch("/api/accounts/{account_id}/positions/{symbol}", response_model=TradePositionOut, tags=["accounts"])
async def update_position(
    account_id: str,
    symbol: str,
    req: TradePositionUpdate,
    user: AuthenticatedUser = Depends(require_user),
):
    async with session_scope(app.state.session_factory) as s:
        repo = TradeAccountRepo(s)
        acc = await repo.get_account(user_id=int(user.id), account_id=str(account_id))
        if not acc:
            raise HTTPException(status_code=404, detail="account not found")
        pos = await repo.get_position(account_pk=acc.id, user_id=int(user.id), symbol=str(symbol).upper())
        if not pos:
            raise HTTPException(status_code=404, detail="position not found")
        pos = await repo.update_position(
            pos,
            shares=req.shares,
            avg_cost_cny=req.avg_cost_cny,
            unrealized_pnl_cny=req.unrealized_pnl_cny,
        )
        return _position_out(pos)

@app.delete("/api/accounts/{account_id}/positions/{symbol}", tags=["accounts"])
async def delete_position(
    account_id: str,
    symbol: str,
    user: AuthenticatedUser = Depends(require_user),
):
    async with session_scope(app.state.session_factory) as s:
        repo = TradeAccountRepo(s)
        acc = await repo.get_account(user_id=int(user.id), account_id=str(account_id))
        if not acc:
            raise HTTPException(status_code=404, detail="account not found")
        deleted = await repo.delete_position(
            account_pk=acc.id,
            user_id=int(user.id),
            symbol=str(symbol).upper(),
        )
        if deleted <= 0:
            raise HTTPException(status_code=404, detail="position not found")
        return {"ok": True, "deleted": deleted, "account_id": account_id, "symbol": str(symbol).upper()}


@app.post("/api/agent/stock-strategy", response_model=StockStrategyAdvice, tags=["agent"])
async def agent_stock_strategy(
    req: StockStrategyRequest,
    user: AuthenticatedUser = Depends(require_user),
):
    # 1) Load account + positions
    async with session_scope(app.state.session_factory) as s:
        repo = TradeAccountRepo(s)
        acc = await repo.get_account(user_id=int(user.id), account_id=str(req.account_id))
        if not acc:
            raise HTTPException(status_code=404, detail="account not found")
        positions = await repo.list_positions(account_pk=acc.id, user_id=int(user.id))

    # 2) Fetch recent bars
    now = now_shanghai()
    end_dt = now.replace(hour=23, minute=59, second=59, microsecond=0)
    start_dt = (end_dt - timedelta(days=int(req.lookback_days) - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start_str = fmt_shanghai(start_dt)
    end_str = fmt_shanghai(end_dt)
    if not start_str or not end_str:
        raise HTTPException(status_code=400, detail="invalid time window")

    symbol = normalize_cn_symbol(req.symbol)
    try:
        if req.bar_type == "daily":
            start_day = start_dt.strftime("%Y%m%d")
            end_day = end_dt.strftime("%Y%m%d")
            df = await run_in_threadpool(
                fetch_cn_daily_bars,
                symbol=symbol,
                start=start_day,
                end=end_day,
            )
        else:
            df = await run_in_threadpool(
                fetch_cn_minute_bars,
                symbol=symbol,
                start=start_str,
                end=end_str,
                freq=req.freq,
                limit=480,
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"stock_bars_error: {e}")

    if df is None or df.empty or "close" not in df.columns:
        raise HTTPException(status_code=400, detail="no price data available")

    closes = df["close"].dropna()
    if closes.empty:
        raise HTTPException(status_code=400, detail="no close prices available")

    first_close = float(closes.iloc[0])
    last_close = float(closes.iloc[-1])
    high = float(df["high"].dropna().max()) if "high" in df.columns else None
    low = float(df["low"].dropna().min()) if "low" in df.columns else None
    change_pct = ((last_close - first_close) / first_close * 100.0) if first_close else 0.0
    returns = closes.pct_change().dropna()
    vol_pct = float(returns.std() * 100.0) if not returns.empty else 0.0

    # 3) Build prompt
    acct_payload = {
        "account_id": str(acc.account_id),
        "cash_cny": float(acc.cash_cny or 0.0),
        "base_currency": str(acc.base_currency or "CNY"),
        "positions": [
            {
                "symbol": str(p.symbol),
                "shares": int(p.shares or 0),
                "avg_cost_cny": float(p.avg_cost_cny) if p.avg_cost_cny is not None else None,
                "unrealized_pnl_cny": float(p.unrealized_pnl_cny) if p.unrealized_pnl_cny is not None else None,
            }
            for p in positions
        ],
    }

    window_label = (
        f"{start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')} (Asia/Shanghai)"
        if req.bar_type == "daily"
        else f"{start_str} ~ {end_str} (Asia/Shanghai)"
    )
    recent_stats = {
        "data_window": window_label,
        "bars": int(len(df)),
        "first_close_cny": first_close,
        "latest_close_cny": last_close,
        "high_cny": high,
        "low_cny": low,
        "change_pct": round(change_pct, 2),
        "volatility_pct": round(vol_pct, 2),
    }

    system_prompt = (
        "You are a stock analyst agent. Provide next-day trading strategy advice "
        "based on recent price action and the user's account context. "
        "Be concise, practical, and risk-aware. Do not invent data."
    )

    user_prompt = (
        "Account:\n"
        f"{acct_payload}\n\n"
        "Recent price summary:\n"
        f"{recent_stats}\n\n"
        f"Target symbol: {symbol}\n"
        f"User note (optional): {req.user_note or ''}\n\n"
        "Return a short summary, a next-day plan (entry/exit/hold guidance), "
        "key levels, risks, and a confidence score between 0 and 1. "
        "Also provide confidence_breakdown with keys buy/hold/sell, each 0..1."
    )

    llm: OpenAICompatLLM = app.state.llm
    try:
        parsed = await llm.chat_json(
            system_prompt,
            user_prompt,
            _strategy_schema_hint(),
            model_cls=StockStrategyLLM,
            temperature=0.4,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"llm_error: {e}")

    chart_points = []
    try:
        tail = df.tail(120)
        for idx, row in tail.iterrows():
            ts = idx.strftime("%Y-%m-%d" if req.bar_type == "daily" else "%Y-%m-%d %H:%M:%S")
            close_v = row.get("close", None)
            if close_v is None:
                continue
            chart_points.append({"ts": ts, "close": float(close_v)})
    except Exception:
        chart_points = []

    return StockStrategyAdvice(
        ok=True,
        account_id=str(acc.account_id),
        symbol=symbol,
        data_window=recent_stats["data_window"],
        latest_close_cny=last_close,
        confidence_breakdown=parsed.confidence_breakdown or {},
        summary=parsed.summary,
        action_bias=parsed.action_bias,
        plan=parsed.plan,
        key_levels=parsed.key_levels,
        risks=parsed.risks,
        confidence=parsed.confidence,
        chart=chart_points,
    )


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


@app.get("/api/stock/daily", tags=["trading"])
async def api_stock_daily(
    symbol: str,
    start: str,
    end: str,
    adjust: str = "",
    user: AuthenticatedUser = Depends(require_user),
):
    code = normalize_cn_symbol(symbol)
    try:
        df = await run_in_threadpool(
            fetch_cn_daily_bars,
            symbol=code,
            start=start,
            end=end,
            adjust=adjust,
        )
        payload = bars_to_payload(df)
        payload.update({
            "symbol": code,
            "start": start,
            "end": end,
            "adjust": adjust,
            "count": len(df),
        })
        return payload
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"stock_daily_error: {e}")


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
