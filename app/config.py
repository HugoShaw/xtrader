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
        validate_default=True,  # 🔥 important
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
    min_confidence: float = Field(default=0.65, alias="MIN_CONFIDENCE")

    # Database
    database_url: str = Field(default="sqlite+aiosqlite:///./xtrader.db", alias="DATABASE_URL") 

settings = Settings()
