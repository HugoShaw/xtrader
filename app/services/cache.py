# app/services/cache.py
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Dict

# ------------------------------------------------------------
# Cache interface (so you can swap Memory <-> Redis later)
# ------------------------------------------------------------

class AsyncCache(Protocol):
    async def get(self, key: str) -> Optional[Any]:
        ...

    async def set(self, key: str, value: Any, ttl_sec: float) -> None:
        ...

    async def delete(self, key: str) -> None:
        ...

    async def close(self) -> None:
        ...


# ------------------------------------------------------------
# In-memory TTL cache (single worker)
# ------------------------------------------------------------

@dataclass
class _Item:
    value: Any
    expires_at: float


class MemoryTTLCache(AsyncCache):
    """
    Simple in-process TTL cache.
    - Best for single Uvicorn worker.
    - Not shared across workers/instances.
    """

    def __init__(self) -> None:
        self._data: Dict[str, _Item] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            it = self._data.get(key)
            if not it:
                return None
            if it.expires_at < time.monotonic():
                self._data.pop(key, None)
                return None
            return it.value

    async def set(self, key: str, value: Any, ttl_sec: float) -> None:
        ttl = float(ttl_sec)
        async with self._lock:
            self._data[key] = _Item(value=value, expires_at=time.monotonic() + ttl)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)

    async def close(self) -> None:
        # nothing to close for memory cache
        return


# ------------------------------------------------------------
# Redis cache placeholder (future)
# ------------------------------------------------------------

class RedisCache(AsyncCache):
    """
    Placeholder for future Redis-based cache.
    You can implement it when you're ready:
      - pip install redis
      - from redis.asyncio import Redis
      - encode values as JSON/msgpack
    """
    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._redis = None  # type: ignore

    async def _ensure(self) -> None:
        if self._redis is not None:
            return
        from redis.asyncio import Redis  # installed later
        # decode_responses=True => strings; if you store bytes, set False
        self._redis = Redis.from_url(self.redis_url, decode_responses=True)

    async def get(self, key: str) -> Optional[Any]:
        await self._ensure()
        raw = await self._redis.get(key)  # type: ignore
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return raw

    async def set(self, key: str, value: Any, ttl_sec: float) -> None:
        await self._ensure()
        try:
            raw = json.dumps(value, ensure_ascii=False)
        except Exception:
            raw = str(value)
        await self._redis.set(key, raw, ex=int(ttl_sec))  # type: ignore

    async def delete(self, key: str) -> None:
        await self._ensure()
        await self._redis.delete(key)  # type: ignore

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.close()
            self._redis = None


# ------------------------------------------------------------
# Factory
# ------------------------------------------------------------

def build_cache(*, backend: str, redis_url: str | None = None) -> AsyncCache:
    backend = (backend or "memory").strip().lower()
    if backend == "memory":
        return MemoryTTLCache()
    if backend == "redis":
        if not redis_url:
            raise ValueError("CACHE_BACKEND=redis requires CACHE_REDIS_URL")
        return RedisCache(redis_url)
    raise ValueError(f"Unknown cache backend: {backend!r}")
