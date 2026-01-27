from fastapi import HTTPException

def normalize_cn_symbol(symbol: str) -> str:
    """
    Normalize A-share symbol:
    - keeps only 6-digit code like '600519', '000001', '301xxx'
    - accepts formats like '600519.SH' -> '600519'
    """
    s = symbol.strip().upper()
    if "." in s:
        s = s.split(".", 1)[0]
    if len(s) != 6 or not s.isdigit():
        raise HTTPException(status_code=400, detail=f"Invalid A-share symbol: {symbol} (expect 6-digit code)")
    return s