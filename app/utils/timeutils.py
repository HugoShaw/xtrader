# app/utils/timeutils.py
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

DEFAULT_TZ_NAME = "Asia/Shanghai"
TS_FMT = "%Y-%m-%d %H:%M:%S"

def tz(tz_name: str = DEFAULT_TZ_NAME) -> ZoneInfo:
    return ZoneInfo(tz_name)

def now_shanghai() -> datetime:
    return datetime.now(tz())

def ensure_shanghai(dt: Optional[datetime]) -> Optional[datetime]:
    """Return tz-aware dt in Asia/Shanghai. If dt is naive, assume it is already Shanghai local."""
    if dt is None:
        return None
    z = tz()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=z)
    return dt.astimezone(z)

def fmt_shanghai(dt: Optional[datetime]) -> Optional[str]:
    """YYYY-MM-DD HH:MM:SS in Shanghai (no offset)."""
    dt2 = ensure_shanghai(dt)
    return dt2.strftime(TS_FMT) if dt2 else None

def parse_shanghai(ts: Optional[str]) -> Optional[datetime]:
    """Parse YYYY-MM-DD HH:MM:SS as Shanghai tz-aware."""
    if not ts:
        return None
    dt = datetime.strptime(ts, TS_FMT)
    return dt.replace(tzinfo=tz())

def must_parse_shanghai(ts: str) -> datetime:
    """Parse or raise a clear error for API boundary inputs."""
    dt = parse_shanghai(ts)
    if dt is None:
        raise ValueError(f"Invalid end_ts format: {ts!r} (expect '{TS_FMT}' in {DEFAULT_TZ_NAME})")
    return dt


def to_shanghai(dt: datetime) -> datetime:
    """Non-optional version of ensure_shanghai()."""
    out = ensure_shanghai(dt)
    assert out is not None
    return out