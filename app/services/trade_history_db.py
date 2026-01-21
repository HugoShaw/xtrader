# app/services/trade_history_db.py
from __future__ import annotations

from datetime import datetime, date
from typing import Any, Dict, List, Optional

from app.storage.db import session_scope
from app.storage.repo import IntradayTradeRepo
from app.utils.timeutils import (
    now_shanghai,
    ensure_shanghai,
    fmt_shanghai,
    parse_shanghai,
)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if (v == v) else None  # NaN guard
    except Exception:
        return None


def _parse_sh_ts(s: Optional[str]) -> Optional[datetime]:
    """Parse Shanghai 'YYYY-MM-DD HH:MM:SS' -> tz-aware datetime."""
    return parse_shanghai(s)


def _as_shanghai_dt(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure tz-aware Shanghai; if naive, assume it's already Shanghai local."""
    if dt is None:
        return None
    return ensure_shanghai(dt)


def _trading_day_sh(now_ts: Optional[str], fallback_dt: Optional[datetime] = None) -> date:
    dt = _parse_sh_ts(now_ts)
    if dt is not None:
        return _as_shanghai_dt(dt).date()

    if fallback_dt is not None:
        return _as_shanghai_dt(fallback_dt).date()

    return now_shanghai().date()


def _normalize_cost_fields_inplace(record: Dict[str, Any]) -> None:
    """
    Option A normalization:
      - total_cost_cny := fee_cny + slippage_cny (if total missing)
      - fees_cny := total_cost_cny (legacy all-in friction)
      - if both present but inconsistent, force fees_cny = total_cost_cny
    """
    fee_cny = _safe_float(record.get("fee_cny"))
    slippage_cny = _safe_float(record.get("slippage_cny"))
    total_cost_cny = _safe_float(record.get("total_cost_cny"))
    fees_cny = _safe_float(record.get("fees_cny"))

    if total_cost_cny is None and (fee_cny is not None or slippage_cny is not None):
        total_cost_cny = float(fee_cny or 0.0) + float(slippage_cny or 0.0)

    if total_cost_cny is not None:
        fees_cny = float(total_cost_cny)

    record["fee_cny"] = fee_cny
    record["slippage_cny"] = slippage_cny
    record["total_cost_cny"] = total_cost_cny
    record["fees_cny"] = fees_cny


class TradeHistoryDB:
    """
    DB-backed intraday trade history used by prompt_builder.
    Stores every decision/execution attempt and provides "today intraday" slices.
    """

    def __init__(self, session_factory, *, max_records: int = 50):
        self.session_factory = session_factory
        self.max_records = int(max_records)

    async def list_intraday_today(
        self,
        symbol: str,
        *,
        now_ts: Optional[str],
        limit: Optional[int] = None,
        include_json: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return Shanghai-today records (chronological) for UI / LLM usage.

        Option A:
          - fees_cny is legacy "all-in friction" and should equal total_cost_cny.
        """
        limit_n = min(int(limit or self.max_records), self.max_records)
        day = _trading_day_sh(now_ts)

        async with session_scope(self.session_factory) as s:
            repo = IntradayTradeRepo(s)
            rows = await repo.list_by_day(symbol, day, limit=limit_n, asc=True)

        out: List[Dict[str, Any]] = []
        for r in rows:
            dt_sh = _as_shanghai_dt(getattr(r, "decision_ts", None))
            ok_raw = getattr(r, "ok", None)

            # pull fields (raw)
            executed_price_cny = _safe_float(getattr(r, "executed_price_cny", None))
            executed_price_net_cny = _safe_float(getattr(r, "executed_price_net_cny", None))
            notional_cny = _safe_float(getattr(r, "notional_cny", None))
            fee_cny = _safe_float(getattr(r, "fee_cny", None))
            slippage_cny = _safe_float(getattr(r, "slippage_cny", None))
            total_cost_cny = _safe_float(getattr(r, "total_cost_cny", None))
            fees_cny = _safe_float(getattr(r, "fees_cny", None))
            cash_delta_cny = _safe_float(getattr(r, "cash_delta_cny", None))
            realized_pnl_cny = _safe_float(getattr(r, "realized_pnl_cny", None))

            # Option A normalization for older rows:
            # 1) derive total_cost if missing but fee/slippage present
            if total_cost_cny is None and (fee_cny is not None or slippage_cny is not None):
                total_cost_cny = float(fee_cny or 0.0) + float(slippage_cny or 0.0)

            # 2) make sure fees_cny mirrors total_cost when available
            if total_cost_cny is not None:
                fees_cny = float(total_cost_cny)

            rec: Dict[str, Any] = {
                "decision_ts": fmt_shanghai(dt_sh) if dt_sh else None,

                "action": getattr(r, "action", None),
                "expected_direction": getattr(r, "expected_direction", None),
                "confidence": _safe_float(getattr(r, "confidence", None)),

                "lot_size": int(getattr(r, "lot_size", 100) or 100),
                "suggested_lots": int(getattr(r, "suggested_lots", 0) or 0),
                "suggested_shares": int(getattr(r, "suggested_shares", 0) or 0),

                "ok": bool(int(ok_raw)) if ok_raw is not None else False,
                "status": getattr(r, "status", None),
                "message": getattr(r, "message", None),

                # existing
                "executed_price_cny": executed_price_cny,
                "fees_cny": fees_cny,  # legacy all-in friction (Option A)
                "realized_pnl_cny": realized_pnl_cny,

                # new (cost-aware)
                "executed_price_net_cny": executed_price_net_cny,
                "notional_cny": notional_cny,
                "fee_cny": fee_cny,
                "slippage_cny": slippage_cny,
                "total_cost_cny": total_cost_cny,
                "cash_delta_cny": cash_delta_cny,

                "reason": getattr(r, "reason", None),
                "risk_notes": getattr(r, "risk_notes", None),
            }

            if include_json:
                rec.update(
                    {
                        "signal_json": getattr(r, "signal_json", None),
                        "snapshot_json": getattr(r, "snapshot_json", None),
                        "broker_details_json": getattr(r, "broker_details_json", None),
                    }
                )

            out.append(rec)

        return out

    async def append_intraday(
        self,
        symbol: str,
        *,
        now_ts: Optional[str],
        record: Dict[str, Any],
    ) -> None:
        """
        Persist a decision/execution attempt row.

        decision_ts is treated as Shanghai local:
          - if str: parse as Shanghai
          - if datetime naive: assume Shanghai
          - if datetime aware: convert to Shanghai then drop tz for SQLite storage
        """
        decision_ts = record.get("decision_ts")
        if isinstance(decision_ts, str):
            dt_sh = _parse_sh_ts(decision_ts)
            if dt_sh is None:
                return
            dt_sh = _as_shanghai_dt(dt_sh)
        elif isinstance(decision_ts, datetime):
            dt_sh = _as_shanghai_dt(decision_ts)
        else:
            return

        # For SQLite datetime columns, store naive local time (Shanghai), consistently.
        decision_dt_naive = dt_sh.replace(tzinfo=None)
        day = _trading_day_sh(now_ts, fallback_dt=dt_sh)

        # ✅ Option A normalization for costs
        _normalize_cost_fields_inplace(record)

        def _safe_int(x: Any, default: int = 0) -> int:
            try:
                if x is None:
                    return int(default)
                return int(x)
            except Exception:
                return int(default)

        row = {
            "symbol": symbol,
            "trading_day_sh": day,
            "decision_ts": decision_dt_naive,

            "action": str(record.get("action", "")),
            "expected_direction": record.get("expected_direction"),
            "confidence": _safe_float(record.get("confidence")),
            "reason": record.get("reason"),
            "risk_notes": record.get("risk_notes"),

            "lot_size": _safe_int(record.get("lot_size"), 100) or 100,
            "suggested_lots": _safe_int(record.get("suggested_lots"), 0),
            "suggested_shares": _safe_int(record.get("suggested_shares"), 0),

            "status": str(record.get("status") or "DECIDED"),
            "ok": 1 if bool(record.get("ok")) else 0,
            "message": record.get("message"),

            # existing
            "executed_price_cny": _safe_float(record.get("executed_price_cny")),
            "fees_cny": _safe_float(record.get("fees_cny")),  # legacy all-in friction (Option A)
            "realized_pnl_cny": _safe_float(record.get("realized_pnl_cny")),

            # new (cost-aware)
            "executed_price_net_cny": _safe_float(record.get("executed_price_net_cny")),
            "notional_cny": _safe_float(record.get("notional_cny")),
            "fee_cny": _safe_float(record.get("fee_cny")),
            "slippage_cny": _safe_float(record.get("slippage_cny")),
            "total_cost_cny": _safe_float(record.get("total_cost_cny")),
            "cash_delta_cny": _safe_float(record.get("cash_delta_cny")),

            "signal_json": record.get("signal_json"),
            "snapshot_json": record.get("snapshot_json"),
            "broker_details_json": record.get("broker_details_json"),
        }

        async with session_scope(self.session_factory) as s:
            repo = IntradayTradeRepo(s)
            await repo.add(row)

    async def clear(self, symbol: str) -> int:
        async with session_scope(self.session_factory) as s:
            repo = IntradayTradeRepo(s)
            return await repo.clear_symbol(symbol)
