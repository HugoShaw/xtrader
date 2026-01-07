from __future__ import annotations

import json
import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, TypeVar

from openai import OpenAI, OpenAIError

from app.models import TradeSignal
from app.logging_config import logger

T = TypeVar("T")


class OpenAICompatLLM:
    """
    OpenAI-compatible LLM client (DeepSeek / Qwen / OpenAI).

    Features:
    - JSON structured output (chat_json)
    - Plain text output (chat_text)
    - Multi-turn chat (chat_messages)
    - Timeout + retry (network + server errors)
    - Latency + token usage logging
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 300.0,
        max_retries: int = 3,
        retry_backoff_s: float = 1.5,
    ):
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=self.timeout_s,
        )

        logger.info(
            f"LLM initialized | base_url={base_url.rstrip('/')} | model={self.model} | "
            f"timeout_s={self.timeout_s} | max_retries={self.max_retries}"
        )

    # ============================================================
    # Internal: extract usage/tokens safely
    # ============================================================
    @staticmethod
    def _usage_dict(resp: Any) -> Dict[str, int]:
        """
        resp.usage may be None or may not exist depending on provider.
        """
        usage = getattr(resp, "usage", None)
        if not usage:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # openai SDK usage typically has these fields
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        tt = int(getattr(usage, "total_tokens", pt + ct) or (pt + ct))
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}

    @staticmethod
    def _request_id(resp: Any) -> str:
        """
        Some providers return request id in response object or headers.
        We keep it best-effort.
        """
        # openai python SDK object sometimes has .id (completion id)
        rid = getattr(resp, "id", None)
        return str(rid) if rid else ""

    @staticmethod
    def _completion_meta(resp: Any) -> Dict[str, Any]:
        """
        Extracts info that helps verify whether the model completed normally.
        """
        try:
            choice0 = resp.choices[0]
        except Exception:
            return {
                "finish_reason": "",
                "choice_index": 0,
                "content_len": 0,
                "content_head": "",
                "content_tail": "",
            }

        finish_reason = getattr(choice0, "finish_reason", "") or ""
        idx = int(getattr(choice0, "index", 0) or 0)

        msg = getattr(choice0, "message", None)
        content = ""
        if msg is not None:
            content = getattr(msg, "content", "") or ""

        # Keep logs safe & small
        head = content[:200].replace("\n", "\\n")
        tail = content[-200:].replace("\n", "\\n") if len(content) > 200 else ""

        return {
            "finish_reason": finish_reason,
            "choice_index": idx,
            "content_len": len(content),
            "content_head": head,
            "content_tail": tail,
        }

    async def _with_retry(
        self,
        call: Callable[[], T],
        *,
        op: str,
    ) -> T:
        last_exc: Optional[Exception] = None

        start_all = time.perf_counter()
        for attempt in range(1, self.max_retries + 1):
            start_attempt = time.perf_counter()
            try:
                resp = await call()
                latency_ms = (time.perf_counter() - start_attempt) * 1000.0

                usage = self._usage_dict(resp)
                rid = self._request_id(resp)
                meta = self._completion_meta(resp)

                logger.info(
                    f"LLM {op} ok | attempt={attempt}/{self.max_retries} | "
                    f"latency_ms={latency_ms:.1f} | "
                    f"prompt_tokens={usage['prompt_tokens']} | "
                    f"completion_tokens={usage['completion_tokens']} | "
                    f"total_tokens={usage['total_tokens']} | "
                    f"finish_reason={meta['finish_reason']} | "
                    f"content_len={meta['content_len']} | "
                    f"model={self.model}"
                    + (f" | request_id={rid}" if rid else "")
                    + (f" | head={meta['content_head']!r}" if meta["content_head"] else "")
                    + (f" | tail={meta['content_tail']!r}" if meta["content_tail"] else "")
                )

                # If finish_reason indicates truncation, log warning (often means "not correct")
                if meta["finish_reason"] and meta["finish_reason"] != "stop":
                    logger.warning(
                        f"LLM {op} non-stop finish_reason={meta['finish_reason']} | "
                        f"model={self.model} | total_tokens={usage['total_tokens']} | "
                        f"hint=output_may_be_truncated_or_filtered"
                    )

                return resp

            except (OpenAIError, TimeoutError, OSError) as e:
                last_exc = e
                latency_ms = (time.perf_counter() - start_attempt) * 1000.0

                logger.warning(
                    f"LLM {op} failed | attempt={attempt}/{self.max_retries} | "
                    f"latency_ms={latency_ms:.1f} | model={self.model} | err={type(e).__name__}: {e}"
                )

                if attempt >= self.max_retries:
                    break

                await asyncio.sleep(self.retry_backoff_s * attempt)

        total_ms = (time.perf_counter() - start_all) * 1000.0
        logger.error(
            f"LLM {op} giving up | attempts={self.max_retries} | total_elapsed_ms={total_ms:.1f} | "
            f"model={self.model} | last_err={type(last_exc).__name__ if last_exc else 'Unknown'}: {last_exc}"
        )
        raise RuntimeError(f"LLM request failed after {self.max_retries} retries: {last_exc}")

    # ============================================================
    # JSON structured output
    # ============================================================
    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_hint: Dict[str, Any],
        *,
        temperature: float = 0.1,
    ) -> TradeSignal:
        """
        Strict JSON output using OpenAI `response_format=json_object`.
        """

        messages = [
            {
                "role": "system",
                "content": (
                    system_prompt
                    + "\n\nYou MUST return valid JSON ONLY.\nSCHEMA:\n"
                    + json.dumps(schema_hint, ensure_ascii=False)
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        async def _call():
            # OpenAI SDK call is sync; run it in a thread to avoid blocking the event loop
            return await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )

        resp = await self._with_retry(_call, op="chat_json")

        content = resp.choices[0].message.content
        try:
            obj = json.loads(content)
            logger.info(
                f"LLM chat_json parsed ok | model={self.model} | keys={sorted(list(obj.keys()))}"
            )

            allowed = set(schema_hint.get("properties", {}).keys())
            extra = set(obj.keys()) - allowed
            if extra:
                raise ValueError(f"LLM returned extra keys not allowed: {sorted(extra)}")
        except json.JSONDecodeError as e:
            # Log parse error with snippet to debug (truncate)
            snippet = (content or "")[:500]
            logger.error(f"LLM chat_json invalid JSON | model={self.model} | content_head={snippet!r}")
            raise ValueError(f"LLM returned invalid JSON: {content}") from e

        return TradeSignal(**obj)

    # ============================================================
    # Plain text chat
    # ============================================================
    async def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        async def _call():
            return await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        resp = await self._with_retry(_call, op="chat_text")
        return resp.choices[0].message.content

    # ============================================================
    # Multi-turn chat
    # ============================================================
    async def chat_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not messages:
            raise ValueError("messages must not be empty")

        async def _call():
            return await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        resp = await self._with_retry(_call, op="chat_messages")
        return resp.choices[0].message.content
