from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class StockBasicInfoOut(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=16)
    asof: Optional[str] = None
    source: str = "stock_individual_info_em"
    data: Dict[str, Any] = Field(default_factory=dict)