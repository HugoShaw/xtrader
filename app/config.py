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
    max_trades_per_day: int = Field(default=10, alias="MAX_TRADES_PER_DAY")
    trade_cooldown_seconds: int = Field(default=180, alias="TRADE_COOLDOWN_SECONDS")
    max_position_value_usd: float = Field(default=5000.0, alias="MAX_POSITION_VALUE_USD")
    fixed_trade_amount_usd: float = Field(default=200.0, alias="FIXED_TRADE_AMOUNT_USD")
    min_confidence: float = Field(default=0.60, alias="MIN_CONFIDENCE")


settings = Settings()
