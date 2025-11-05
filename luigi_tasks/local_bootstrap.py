
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier

ART = Path("artifacts"); ART.mkdir(exist_ok=True, parents=True)
FEATURES = ["ret1","gap","ma7","ma14","atr","volchg","ret5","ret10","ma_spread_7_14"]

def _load_all_csvs():
    data_dir = Path("data/uploads")
    dfs = []
    for p in data_dir.glob("*.csv"):
        df = pd.read_csv(p)
        if "Symbol" not in df.columns: df["Symbol"] = p.stem.upper()
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No CSVs found in data/uploads. Upload first.")
    return pd.concat(dfs, ignore_index=True)

def _engineer(df: pd.DataFrame):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol","Date"]).reset_index(drop=True)
    out = []
    for sym, g in df.groupby("Symbol", sort=False):
        g = g.copy()
        g["ret1"] = g["Close"].pct_change()
        g["gap"] = (g["Open"] - g["Close"].shift(1)) / g["Close"].shift(1)
        g["ma7"] = g["Close"].rolling(7).mean()
        g["ma14"] = g["Close"].rolling(14).mean()
        g["atr"] = (g["High"]-g["Low"]).rolling(7).mean()
        g["volchg"] = g["Volume"].pct_change()
        g["ret5"] = g["Close"].pct_change(5)
        g["ret10"] = g["Close"].pct_change(10)
        g["ma_spread_7_14"] = (g["ma7"] - g["ma14"]) / g["ma14"]
        g["target"] = (g["Close"].shift(-1) > g["Close"]).astype(int)
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df.dropna(subset=FEATURES+["target"], inplace=True)
    return df

def _train_per_symbol(df: pd.DataFrame):
    symbols = sorted(df["Symbol"].unique())
    results = {}
    for s in symbols:
        d = df[df["Symbol"] == s].copy()
        X = d[FEATURES]
        var = X.var(numeric_only=True)
        keep_feats = var[var > 1e-12].index.tolist() or FEATURES
        y = d["target"].astype(int)
        split = int(len(d)*0.7)
        Xtr, Xte = d[keep_feats].iloc[:split], d[keep_feats].iloc[split:]
        ytr, yte = y.iloc[:split], y.iloc[split:]
        clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, min_child_samples=1, random_state=42)
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        prob = clf.predict_proba(Xte)[:,1]
        acc = float(accuracy_score(yte, yp)) if len(yte.unique())>1 else 0.5
        try:
            auc = float(roc_auc_score(yte, prob)) if len(yte.unique())>1 else 0.5
        except Exception:
            auc = 0.5
        joblib.dump(clf, ART/f"model_{s}.pkl")
        (ART/f"features_{s}.json").write_text(json.dumps(keep_feats))
        pd.DataFrame({"Date": d["Date"].iloc[split:].to_list(), **{f: Xte[f].to_list() for f in keep_feats},
                      "p_up": prob, "pred": yp, "target": yte.to_list()}).to_csv(ART/f"preds_{s}.csv", index=False)
        results[s] = {"acc": acc, "auc": auc, "features": keep_feats, "rows": int(len(d))}
    ranks = []
    for s in results:
        preds = pd.read_csv(ART/f"preds_{s}.csv")
        if len(preds)==0: continue
        last = preds.iloc[-1]
        ranks.append({"symbol": s, "p_up": float(last["p_up"]), "acc": results[s]["acc"], "auc": results[s]["auc"]})
    pd.DataFrame(ranks).sort_values("p_up", ascending=False).to_csv(ART/"rankings.csv", index=False)
    return {"symbols": results, "ranking_file": str(ART/"rankings.csv")}

def bootstrap_all():
    raw = _load_all_csvs()
    df = _engineer(raw)
    info = _train_per_symbol(df)
    return {"status":"ok", **info}

def predict_on_date(date: str, symbol: str|None):
    out = {}
    for p in Path("artifacts").glob("preds_*.csv"):
        s = p.stem.replace("preds_","")
        pdf = pd.read_csv(p)
        pdf["Date"] = pd.to_datetime(pdf["Date"])
        qd = pd.to_datetime(date)
        pdf2 = pdf[pdf["Date"]<=qd]
        if len(pdf2)==0: continue
        last = pdf2.iloc[-1]
        if symbol and s.upper()!=symbol.upper(): continue
        out[s] = {"date": str(last["Date"]), "p_up": float(last["p_up"]), "pred": int(last["pred"])}
    return {"query_date": date, "predictions": out}
