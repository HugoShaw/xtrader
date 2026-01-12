# xtrader

**xtrader** is an intraday quantitative trading assistant for China A-shares powered by LLMs (DeepSeek/Qwen/OpenAI-compatible).
It combines **real-time market data**, **account state**, **risk control**, and **trade history feedback** to generate disciplined trading signals every 30 minutes.

> ⚠️ This project is for **research & paper trading** only.
> Do NOT use with real capital without extensive testing.

---

## Features

* 🔹 LLM-driven trade decision engine (BUY / SELL / HOLD)
* 🔹 Uses **AkShare (Eastmoney)** real-time & minute bars
* 🔹 Account-aware decisions (cash, position, cost, PnL)
* 🔹 Strict **risk management**

  * max trades per day
  * cooldown
  * confidence threshold
  * max position exposure
* 🔹 Trade history stored in **SQLite**
* 🔹 Paper broker (no real orders)
* 🔹 Backtest UI & API
* 🔹 Direct LLM chat endpoint (structured JSON supported)
* 🔹 Full logging & observability

---

## Architecture

```
xtrader/
│
├── app/
│   ├── main.py                # FastAPI entrypoint
│   ├── config.py              # Settings (.env)
│   ├── logging_config.py
│
│   ├── services/
│   │   ├── strategy.py        # Core trading engine
│   │   ├── prompt_builder.py # LLM prompt
│   │   ├── market_data.py
│   │   ├── market_data_akshare.py
│   │   ├── risk.py
│   │   ├── broker.py
│   │   ├── llm_client.py
│   │   └── trade_history_db.py
│   │   └── cache.py
│
│   ├── storage/
│   │   ├── db.py              # Async SQLite
│   │   ├── orm_models.py
│   │   └── repo.py
│
│   ├── models.py
│   └── models_llm.py
│
├── static/
│   └── backtest.html
│
├── .env
└── README.md
```

---

## Tech Stack

* **FastAPI**
* **SQLAlchemy (async)**
* **SQLite (production mode, WAL)**
* **AkShare**
* **DeepSeek / OpenAI compatible LLM**
* **Pydantic v2**
* **Python 3.10+**

---

## Installation

```bash
git clone <your_repo>
cd xtrader

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Environment Variables (.env)

```env
# Market
MARKET_PROVIDER=akshare

# LLM
LLM_PROVIDER=deepseek
LLM_API_KEY=sk-xxxx
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-reasoner

# Risk guardrails
MAX_TRADES_PER_DAY=5
TRADE_COOLDOWN_SECONDS=1200
MAX_POSITION_VALUE_CNY=100000
FIXED_TRADE_AMOUNT_CNY=5000
MIN_CONFIDENCE=0.65

# Database
DATABASE_URL=sqlite+aiosqlite:///./xtrader.db
```

---

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open:

* Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## Core Endpoints

### 1) Generate Signal

```http
POST /signal/{symbol}
```

Example:

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

Response:

```json
{
  "action": "HOLD",
  "horizon_minutes": 30,
  "confidence": 0.42,
  "expected_direction": "FLAT",
  "suggested_notional_usd": 0,
  "reason": "...",
  "risk_notes": "..."
}
```

---

### 2) Execute (Paper Trade)

```http
POST /execute/{symbol}
```

* Runs signal
* Applies risk checks
* Places **paper order**
* Stores execution & feedback

---

### 3) Trade History

```http
GET /trade_history/{symbol}
POST /trade_history/{symbol}/clear
```

---

### 4) Stock Bars

```http
GET /api/stock/bars?symbol=000100&start=2026-01-07 09:30:00&end=2026-01-07 15:00:00&freq=5
```

---

### 5) LLM Passthrough

```http
POST /api/llm/chat
```

Supports:

* plain text chat
* JSON schema structured output

---

### 6) Backtest

* UI: `/backtest`
* API:

  * `/api/backtest/run`
  * `/api/backtest/report`

---

## Trading Logic

1. Fetch snapshot (AkShare)
2. Load last N trade history
3. Build structured prompt:

   * OHLCV bars
   * account state
   * constraints
4. LLM outputs JSON signal
5. RiskManager validates:

   * trade count
   * cooldown
   * confidence
   * exposure
   * cash / position
6. Broker executes paper trade
7. Persist feedback to DB

---

## Risk Controls

* Max trades per day
* Cooldown between trades
* Confidence threshold
* Max portfolio exposure
* Lot size enforcement
* Cash / position validation

---

## Logging

All key events are logged:

* signal_start
* snapshot fetch time
* bars count
* LLM latency
* risk blocks
* execution result
* DB persistence errors

Example:

```
signal_done | symbol=000100 action=HOLD conf=0.42 bars=12 hist=3 ms_llm=32000
```

---

## Database

* SQLite (async)
* WAL enabled
* Tables:

  * executions
  * trade_feedback
* Auto-created on startup

---

## Safety Notes

* No real broker integration
* LLM decisions are probabilistic
* Always test with paper trading
* Never trust a single model output

---

## Roadmap

* [ ] Real broker API
* [ ] Multi-symbol portfolio
* [ ] Reinforcement learning feedback loop
* [ ] Strategy ensembles
* [ ] Web dashboard
* [ ] Live PnL tracking
* [ ] Alert system

---

## Disclaimer

This software is provided **as-is**.
You are solely responsible for any trading decisions.

--- 