# app/services/llm_client.py
from __future__ import annotations
import httpx
import json
from typing import Any, Dict, List
from app.models import TradeSignal

class OpenAICompatLLM:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    async def chat_json(self, system_prompt: str, user_prompt: str, schema_hint: Dict[str, Any]) -> TradeSignal:
        """
        Calls /chat/completions (OpenAI-compatible).
        Expects the model to return JSON only.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt + "\nSCHEMA:\n" + json.dumps(schema_hint)},
            {"role": "user", "content": user_prompt},
        ]

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
        }

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        # Strict JSON parse
        obj = json.loads(content)
        return TradeSignal(**obj)
