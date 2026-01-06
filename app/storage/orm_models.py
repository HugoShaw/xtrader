# app/storage/orm_models.py
from __future__ import annotations

from datetime import datetime
from typing import Optional, Any, Dict

from sqlalchemy import (
    String,
    Integer,
    Float,
    DateTime,
    Text,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.storage.db import Base


class TradeFeedbackORM(Base):
    """
    Stores 'trade_history.records' items used as LLM feedback.
    """
    __tablename__ = "trade_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    symbol: Mapped[str] = mapped_column(String(16), index=True)
    decision_ts: Mapped[datetime] = mapped_column(DateTime, index=True)

    action: Mapped[str] = mapped_column(String(8))  # BUY/SELL/HOLD
    expected_direction: Mapped[str] = mapped_column(String(8))  # UP/DOWN/FLAT
    confidence: Mapped[float] = mapped_column(Float)

    executed_price_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    shares: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    fees_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    realized_pnl_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


Index("ix_trade_feedback_symbol_decision_ts", TradeFeedbackORM.symbol, TradeFeedbackORM.decision_ts)


class ExecutionORM(Base):
    """
    Stores each /execute result (paper or real later).
    Keep raw JSON blobs as TEXT for easy evolution without migrations.
    """
    __tablename__ = "executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    symbol: Mapped[str] = mapped_column(String(16), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)

    action: Mapped[str] = mapped_column(String(8))  # BUY/SELL/HOLD
    notional_usd: Mapped[float] = mapped_column(Float, default=0.0)
    paper: Mapped[int] = mapped_column(Integer, default=1)  # 1/0

    ok: Mapped[int] = mapped_column(Integer, default=0)  # 1/0
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # JSON text fields (store dict dumps)
    signal_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    snapshot_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    broker_details_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


Index("ix_executions_symbol_ts", ExecutionORM.symbol, ExecutionORM.ts)
