
import io, json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request, Query
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

from api.llm_client import explain
from xai.extra import run_lime as xai_lime, run_anchors as xai_anchors, run_dice as xai_dice, run_ebm as xai_ebm
from luigi_tasks.local_bootstrap import bootstrap_all, predict_on_date as _predict_on_date

app = FastAPI()
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

@app.get("/", response_class=HTMLResponse)
def root():
    return (Path("ui/index.html")).read_text()

@app.post("/upload_csv")
def upload_csv(file: UploadFile = File(...)):
    content = file.file.read()
    df = pd.read_csv(io.BytesIO(content))
    if "Date" not in df.columns:
        return PlainTextResponse("Missing 'Date' column", status_code=400)
    name = Path(file.filename).stem.upper()
    if "Symbol" not in df.columns or df["Symbol"].isna().all():
        df["Symbol"] = name
    symbol = str(df["Symbol"].dropna().iloc[0]).upper()
    outp = Path("data/uploads")/f"{symbol}.csv"
    df.to_csv(outp, index=False)
    return {"status":"ok","saved":str(outp),"rows":int(len(df)),"symbol":symbol}

@app.get('/inspect/features_and_normalization')
def inspect_features_and_normalization():
    feats = ["ret1","gap","ma7","ma14","atr","volchg","ret5","ret10","ma_spread_7_14"]
    normalization = {
        "cleaning": "Coerce non-numeric→NaN; replace Inf; drop rows missing feats/target.",
        "per_symbol_variance_guard": "Drop ~constant features per symbol.",
        "scaling": "No scaling for tree models."
    }
    return {"features": feats, "normalization": normalization}

@app.post("/bootstrap")
def bootstrap():
    return bootstrap_all()

@app.get("/predict_on_date")
def predict_on_date(date: str, symbol: str|None = None):
    return _predict_on_date(date, symbol)

@app.post("/llm/explain")
async def llm_explain(req: Request, model: str = Query(default=None)):
    body = await req.json()
    kind = body.get("kind","data_profile")
    symbol = body.get("symbol")
    ctx = {"features": ["ret1","gap","ma7","ma14","atr","volchg","ret5","ret10","ma_spread_7_14"]}
    text = explain(kind, symbol, ctx, model_override=model)
    return {"kind": kind, "symbol": symbol, "model": model or "env/default", "text": text}

@app.get("/artifacts/{path:path}")
def serve_artifacts(path: str):
    p = Path("artifacts")/path
    if not p.exists():
        return PlainTextResponse("Not found", status_code=404)
    return FileResponse(p)

@app.get("/xai/lime/{symbol}")
def xai_run_lime(symbol: str):
    return xai_lime(symbol.upper())

@app.get("/xai/anchors/{symbol}")
def xai_run_anchors(symbol: str):
    return xai_anchors(symbol.upper())

@app.get("/xai/dice/{symbol}")
def xai_run_dice(symbol: str, total_CFs: int = 3, desired_class: int = 1):
    return xai_dice(symbol.upper(), total_CFs=total_CFs, desired_class=desired_class)

@app.get("/xai/ebm/{symbol}")
def xai_run_ebm(symbol: str):
    return xai_ebm(symbol.upper())
