
import os
from typing import Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

FALLBACK_TEXTS = {
    "data_profile": "Daily OHLCV time series per symbol. Schema checks, type coercion, NaN/Inf handling.",
    "features": "ret1, gap, ma7, ma14, atr, volchg, ret5, ret10, ma_spread_7_14 engineered per symbol.",
    "normalization": "Replace Inf→NaN; drop incomplete rows; per‑symbol variance guard; no scaling for tree models.",
    "ranking": "Predict next‑day up‑move P(up); rank symbols by latest P(up) adjusted for recent volatility.",
    "xai_summary": "TreeSHAP/Perm/PDP global; LIME local; Anchors rule‑style; DiCE counterfactuals; EBM global additive."
}

def _fallback(kind: str, symbol: Optional[str], context: dict) -> str:
    return FALLBACK_TEXTS.get(kind, "Explanation not available.")

def explain(kind: str, symbol: Optional[str], context: dict, model_override: Optional[str] = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = model_override or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key or OpenAI is None:
        return _fallback(kind, symbol, context)
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        sys_prompt = ("You are an XAI assistant for a stock OHLCV pipeline. "
                      "Explain concisely and tie back to engineered features & XAI methods.")
        user_prompt = f"Kind: {kind}\nSymbol: {symbol or 'ALL'}\nContext: {context}\nReturn a short, presentation‑ready paragraph."
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_prompt},
                      {"role":"user","content":user_prompt}],
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return _fallback(kind, symbol, context)
