
# Luigi XAI Stocks (v3) — OpenAI‑wired

Batch OHLCV CSV → Features → LightGBM → t+1 prediction → Ranking → XAI (TreeSHAP, Perm, PDP, LIME, Anchors, DiCE, EBM) → LLM explanations with picker + fallback.

## Quickstart
```bash
unzip luigi_xai_stocks_app_v3.zip -d .
cd luigi_xai_stocks_app_v3
bash run.sh
# open http://localhost:8000
```

## Enable OpenAI (optional)
```bash
export OPENAI_API_KEY="sk-proj-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
```
If not set or on quota errors, the server auto‑falls back to a local explainer.

## CSV schema
`Date, Open, High, Low, Close, Volume[, Symbol]` (Symbol inferred from filename if missing).

## Flow
Upload → Bootstrap → Predict date → LLM Explain → XAI Suite per symbol (LIME/Anchors/DiCE/EBM).
