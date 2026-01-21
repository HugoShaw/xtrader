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
)
from sqlalchemy.orm import Mapped, mapped_column

from app.storage.db import Base


class IntradayTradeORM(Base):
    __tablename__ = "intraday_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

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
