# xtrader

xtrader is an intraday quantitative trading assistant for China A-shares powered by OpenAI-compatible LLMs (DeepSeek/Qwen/OpenAI, etc.). It combines real-time market data, account state, risk controls, trade-history feedback, and a simple cookie-based authentication system to generate disciplined 30-minute trading signals.

> This project is for research and paper trading only. Do not use real capital without extensive testing.

---

## Project Structure

```text
xtrader/
  app/
    main.py                # FastAPI entrypoint and routes
    config.py              # Settings loaded from .env (Pydantic v2)
    logging_config.py
    models.py              # Core trading models and API schemas
    models_llm.py          # LLM chat/request schemas
    models_auth.py         # Auth request/response schemas
    services/
      strategy.py          # StrategyEngine: signal + execution flow
      prompt_builder.py    # System prompt + structured user prompt
      auth.py              # Password hashing + signed cookie sessions
      risk.py              # RiskManager guardrails
      broker.py            # Paper broker (no real orders)
      market_data.py       # Market data interface + provider factory
      market_data_akshare.py
      trade_history_db.py  # DB-backed intraday history feedback loop
      backtest.py          # Backtest API helpers and report HTML
      llm_client.py        # OpenAI-compatible LLM client
      stock_api.py         # Stock bar endpoints/helpers
      cache.py             # Cache abstraction (memory/redis)
    storage/
      db.py                # Async SQLAlchemy engine/session/init
      orm_models.py        # ORM table definitions (includes users table)
      auth_repo.py         # User repository
      repo.py              # Query and persistence layer
    utils/
      timeutils.py
      textutils.py
  static/
    signup.html
    login.html
    signal.html
    execute.html
    backtest.html
  scripts/
    migrate_intraday_trades_add_costs.py
  .env
  README.md
```

---

## Tech Stack

- FastAPI
- SQLAlchemy (async) + aiosqlite
- SQLite (WAL)
- AkShare (Eastmoney)
- OpenAI-compatible LLM APIs
- Pydantic v2
- Python 3.10+

---

## Quick Start

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Configure `.env` (see below).

4. Run the API.

```powershell
uvicorn app.main:app --reload --port 8000
```

---

## Environment Variables (.env)

These map directly to `app/config.py`.

```env
# Market
MARKET_PROVIDER=akshare
MARKET_API_KEY=

# LLM (OpenAI-compatible)
LLM_PROVIDER=deepseek
LLM_API_KEY=sk-xxxx
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# Risk / execution guardrails
MAX_TRADES_PER_DAY=5
TRADE_COOLDOWN_SECONDS=1200
MAX_POSITION_VALUE_CNY=100000
FIXED_TRADE_AMOUNT_CNY=5000
FIXED_SLOTS=1
MIN_CONFIDENCE=0.65

# Cost model
BUY_FEE_RATE=0.00015
SELL_FEE_RATE=0.0025

# Database
DATABASE_URL=sqlite+aiosqlite:///./xtrader.db

# Cache
CACHE_BACKEND=memory
CACHE_REDIS_URL=redis://localhost:6379/0
CACHE_SPOT_TTL_SEC=3
CACHE_ORDERBOOK_TTL_SEC=1
CACHE_BARS_TTL_SEC=2

# Timezone
TIMEZONE_NAME=Asia/Shanghai

# Auth (cookie session)
AUTH_SECRET_KEY=change-this-in-prod
AUTH_SESSION_TTL_SECONDS=43200
AUTH_COOKIE_NAME=xtrader_session
```

---

## Authentication

The app now uses a simple cookie-based session for protecting most routes.

Public pages:
- `GET /signup`
- `GET /login`

Auth APIs:
- `POST /auth/signup`
- `POST /auth/login`
- `POST /auth/logout`
- `GET /auth/me`

Behavior:
- On successful login, the backend sets an HTTP-only cookie.
- Protected routes require that cookie (no explicit `user` query/body param is needed).
- The `execute.html` page includes a quick-login modal and automatically checks `/auth/me`.

---

## Core Endpoints

Public:
- `GET /health`
- `GET /signup`
- `GET /login`
- `POST /auth/signup`
- `POST /auth/login`
- `POST /auth/logout`
- `GET /auth/me` (returns 401 if not logged in)

Protected (requires login cookie):
- `GET /risk`
- `GET /signal-ui`
- `POST /signal/{symbol}`
- `POST /execute/{symbol}`
- `GET /trade_history/{symbol}`
- `POST /trade_history/{symbol}/clear`
- `GET /api/stock/bars`
- `POST /api/llm/chat`
- `GET /backtest`
- `POST /api/backtest/run`
- `POST /api/backtest/report`

Swagger UI is available at `/docs` after you start the server.

---

## Signal Request Example

```http
POST /signal/{symbol}
```

```json
{
  "now_ts": "2026-01-07 15:00:00",
  "account_state": {
    "cash_cny": 100000,
    "position_shares": 10000,
    "avg_cost_cny": 4.8,
    "unrealized_pnl_cny": -900
  }
}
```

Example response:

```json
{
  "action": "HOLD",
  "horizon_minutes": 30,
  "confidence": 0.42,
  "expected_direction": "FLAT",
  "suggested_lots": 0,
  "reason": "...",
  "risk_notes": "..."
}
```

---

## How It Works (High Level)

1. Authenticate once via `/login` or `/auth/login`.
2. Fetch a market snapshot and recent bars.
3. Load today’s intraday trade history from SQLite.
4. Build a structured, fee-aware prompt (`prompt_builder.py`).
5. Ask the LLM for strict JSON output.
6. Apply risk gating (`risk.py`).
7. Execute a paper order (`broker.py`).
8. Persist an intraday record for feedback (`trade_history_db.py` + `storage/*`).

---

## Logging

Key events are logged, including snapshot fetch time, bar counts, LLM latency, risk blocks, execution results, and DB persistence errors.

Example:

```text
signal_done | symbol=000100 action=HOLD conf=0.42 bars=12 hist=3 ms_llm=32000
```

---

## Database Notes

- SQLite with async SQLAlchemy
- WAL pragmas enabled on startup
- Schema auto-initialized in the app lifespan
- Includes a `users` table for authentication
- Primary persistence path: `TradeHistoryDB` -> `storage/repo.py`

---

## Safety Notes

- There is no real broker integration.
- LLM decisions are probabilistic.
- Always validate behavior with paper trading and backtests.

---

## Roadmap Ideas

- Real broker API integration
- Multi-symbol portfolio management
- Strategy ensembles
- Live PnL tracking and alerts
- Web dashboard improvements
