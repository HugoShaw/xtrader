# app/storage/orm_models.py
from __future__ import annotations

from datetime import datetime, date
from typing import Optional

from sqlalchemy import (
    String,
    Integer,
    Float,
    DateTime,
    Date,
    Text,
    Index,
    ForeignKey,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.storage.db import Base


class UserORM(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    salt: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class IntradayTradeORM(Base):
    __tablename__ = "intraday_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    user_id: Mapped[int] = mapped_column(Integer, index=True, default=0)
    username: Mapped[str] = mapped_column(String(64), index=True, default="")
    account_id: Mapped[str] = mapped_column(String(64), index=True, default="default")

    symbol: Mapped[str] = mapped_column(String(16), index=True)
    trading_day_sh: Mapped[date] = mapped_column(Date, index=True)
    decision_ts: Mapped[datetime] = mapped_column(DateTime, index=True)

    action: Mapped[str] = mapped_column(String(8))  # BUY/SELL/HOLD
    expected_direction: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    risk_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    lot_size: Mapped[int] = mapped_column(Integer, default=100)
    suggested_lots: Mapped[int] = mapped_column(Integer, default=0)
    suggested_shares: Mapped[int] = mapped_column(Integer, default=0)

    status: Mapped[str] = mapped_column(String(16), default="DECIDED")
    ok: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # execution (existing)
    executed_price_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fees_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    realized_pnl_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ✅ NEW: cost-aware execution fields (nullable, backward compatible)
    executed_price_net_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    notional_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fee_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    slippage_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_cost_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cash_delta_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    signal_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    snapshot_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    broker_details_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


Index("ix_intraday_trades_symbol_day_ts", IntradayTradeORM.symbol, IntradayTradeORM.trading_day_sh, IntradayTradeORM.decision_ts)
Index("ix_intraday_trades_symbol_ts", IntradayTradeORM.symbol, IntradayTradeORM.decision_ts)
Index(
    "ix_intraday_trades_user_account_symbol_day_ts",
    IntradayTradeORM.user_id,
    IntradayTradeORM.account_id,
    IntradayTradeORM.symbol,
    IntradayTradeORM.trading_day_sh,
    IntradayTradeORM.decision_ts,
)


class ApiUsageORM(Base):
    __tablename__ = "api_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(128), index=True)
    method: Mapped[str] = mapped_column(String(8))
    status_code: Mapped[int] = mapped_column(Integer)
    username: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


Index("ix_api_usage_day_path", ApiUsageORM.created_at, ApiUsageORM.path)


class TradeAccountORM(Base):
    __tablename__ = "trade_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True, default=0)
    account_id: Mapped[str] = mapped_column(String(64), index=True, default="default")
    name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    base_currency: Mapped[str] = mapped_column(String(8), default="CNY")
    cash_cny: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


Index("ix_trade_accounts_user_account", TradeAccountORM.user_id, TradeAccountORM.account_id, unique=True)


class TradePositionORM(Base):
    __tablename__ = "trade_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_pk: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("trade_accounts.id", ondelete="CASCADE"),
        index=True,
    )
    user_id: Mapped[int] = mapped_column(Integer, index=True, default=0)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    shares: Mapped[int] = mapped_column(Integer, default=0)
    avg_cost_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unrealized_pnl_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


Index("ix_trade_positions_account_symbol", TradePositionORM.account_pk, TradePositionORM.symbol, unique=True)
Index("ix_trade_positions_user_symbol", TradePositionORM.user_id, TradePositionORM.symbol)


class StockBarORM(Base):
    __tablename__ = "stock_bars"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    period: Mapped[str] = mapped_column(String(16), index=True)
    adjust: Mapped[str] = mapped_column(String(8), default="")
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)

    open: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    close: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vwap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    amplitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pct_change: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    change: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    turnover: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


Index("ix_stock_bars_symbol_period", StockBarORM.symbol, StockBarORM.period)
Index(
    "ix_stock_bars_symbol_period_adjust_ts",
    StockBarORM.symbol,
    StockBarORM.period,
    StockBarORM.adjust,
    StockBarORM.ts,
    unique=True,
)


class StockBarFetchORM(Base):
    __tablename__ = "stock_bar_fetches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    period: Mapped[str] = mapped_column(String(16), index=True)
    adjust: Mapped[str] = mapped_column(String(8), default="")
    last_fetch_date: Mapped[date] = mapped_column(Date, index=True)
    last_fetch_start: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    last_fetch_end: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


Index(
    "ix_stock_bar_fetch_symbol_period_adjust",
    StockBarFetchORM.symbol,
    StockBarFetchORM.period,
    StockBarFetchORM.adjust,
    unique=True,
)
