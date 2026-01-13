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
    """
    One row per strategy decision/execution attempt.
    Used both for audit and as LLM feedback (intraday-only).
    """
    __tablename__ = "intraday_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    symbol: Mapped[str] = mapped_column(String(16), index=True)

    # ✅ Shanghai trading day for fast "today" queries
    trading_day_sh: Mapped[date] = mapped_column(Date, index=True)

    # decision time
    decision_ts: Mapped[datetime] = mapped_column(DateTime, index=True)

    # signal fields
    action: Mapped[str] = mapped_column(String(8))  # BUY/SELL/HOLD
    expected_direction: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)  # UP/DOWN/FLAT
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    risk_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # sizing
    lot_size: Mapped[int] = mapped_column(Integer, default=100)
    suggested_lots: Mapped[int] = mapped_column(Integer, default=0)
    suggested_shares: Mapped[int] = mapped_column(Integer, default=0)

    # outcome
    status: Mapped[str] = mapped_column(String(16), default="DECIDED")
    # DECIDED | HOLD | BLOCKED | EXECUTED | FAILED

    ok: Mapped[int] = mapped_column(Integer, default=0)  # 1/0
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # execution fields (optional / future)
    executed_price_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fees_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    realized_pnl_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # raw JSON (future-proof)
    signal_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    snapshot_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    broker_details_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


Index("ix_intraday_trades_symbol_day_ts", IntradayTradeORM.symbol, IntradayTradeORM.trading_day_sh, IntradayTradeORM.decision_ts)
Index("ix_intraday_trades_symbol_ts", IntradayTradeORM.symbol, IntradayTradeORM.decision_ts)
