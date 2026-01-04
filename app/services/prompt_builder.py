# app/services/prompt_builder.py
from __future__ import annotations
from app.models import MarketSnapshot

SYSTEM_PROMPT = """You are a cautious quantitative trading assistant.
You must output ONLY valid JSON that matches the required schema.
Do not include markdown. Do not include extra keys.
If data is insufficient, choose HOLD with low confidence.
"""

def build_user_prompt(snapshot: MarketSnapshot, *, fixed_amount: float, max_trades_left: int) -> str:
    # You can add your own derived features here: returns, vol, ATR, RSI, etc.
    payload = {
        "task": "30-minute directional assessment and trading signal",
        "constraints": {
            "fixed_trade_amount_usd": fixed_amount,
            "max_trades_left_today": max_trades_left,
            "time_horizon_minutes": 30,
            "prefer_low_volatility": True,
            "action_rule": "BUY fixed amount if you expect DOWN in next 30m; SELL fixed amount if you expect UP in next 30m; else HOLD"
        },
        "market_snapshot": snapshot.model_dump()
    }
    return str(payload)
