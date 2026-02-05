# app/config.py
from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]  # /xtrader
ENV_PATH = BASE_DIR / ".env"


class Settings(BaseSettings):
    # Pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    # Market data provider
    market_provider: str = Field(default="akshare", alias="MARKET_PROVIDER")
    market_api_key: str = Field(default="", alias="MARKET_API_KEY")

    # LLM provider (OpenAI-compatible)
    llm_provider: str = Field(default="deepseek", alias="LLM_PROVIDER")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="https://api.deepseek.com/v1", alias="LLM_BASE_URL")
    llm_model: str = Field(default="deepseek-chat", alias="LLM_MODEL")

    # Trading guardrails
    max_trades_per_day: int = Field(default=5, alias="MAX_TRADES_PER_DAY")
    trade_cooldown_seconds: int = Field(default=1200, alias="TRADE_COOLDOWN_SECONDS")
    max_position_value_cny: float = Field(default=100000.0, alias="MAX_POSITION_VALUE_CNY")
    fixed_trade_amount_cny: float = Field(default=5000.0, alias="FIXED_TRADE_AMOUNT_CNY")
    fixed_slots: float = Field(default=1, alias="FIXED_SLOTS")
    min_confidence: float = Field(default=0.65, alias="MIN_CONFIDENCE")

    buy_fee_rate: float = Field(default=0.00015, alias="BUY_FEE_RATE")
    sell_fee_rate: float = Field(default=0.0025, alias="SELL_FEE_RATE")

    # Database
    database_url: str = Field(default="sqlite+aiosqlite:///./xtrader.db", alias="DATABASE_URL") 

    cache_backend: str = Field(default="memory", alias="CACHE_BACKEND")  # memory | redis
    cache_redis_url: str = Field(default="redis://localhost:6379/0", alias="CACHE_REDIS_URL")

    # recommended TTLs for your market data
    cache_spot_ttl_sec: float = Field(default=3.0, alias="CACHE_SPOT_TTL_SEC")
    cache_orderbook_ttl_sec: float = Field(default=1.0, alias="CACHE_ORDERBOOK_TTL_SEC")
    cache_bars_ttl_sec: float = Field(default=2.0, alias="CACHE_BARS_TTL_SEC")
 
    timezone_name: str = Field(default="Asia/Shanghai", alias="TIMEZONE_NAME")

    # Stock API fallback metrics
    stock_api_fallback_log: bool = Field(default=False, alias="STOCK_API_FALLBACK_LOG")

    # Auth (simple cookie session)
    auth_secret_key: str = Field(default="dev-insecure-change-me", alias="AUTH_SECRET_KEY")
    auth_session_ttl_seconds: int = Field(default=60 * 60 * 12, alias="AUTH_SESSION_TTL_SECONDS")
    auth_cookie_name: str = Field(default="xtrader_session", alias="AUTH_COOKIE_NAME")

    # Trade history
    trade_history_max_records: int = Field(default=200, alias="TRADE_HISTORY_MAX_RECORDS")
    trade_history_prompt_limit: int = Field(default=30, alias="TRADE_HISTORY_PROMPT_LIMIT")

    # Admin (comma-separated usernames)
    admin_usernames: str = Field(default="admin", alias="ADMIN_USERNAMES")

settings = Settings()
