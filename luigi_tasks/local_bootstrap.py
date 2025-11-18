from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier, LGBMRegressor
import matplotlib.pyplot as plt


# === Directories ===
ART = Path("artifacts"); ART.mkdir(exist_ok=True, parents=True)
ART_FI = ART / "feature_importance"; ART_FI.mkdir(exist_ok=True, parents=True)

FEATURES = ["ret1","gap","ma7","ma14","atr","volchg","ret5","ret10","ma_spread_7_14"]


# === Helper: safe float for JSON (NaN/inf -> None) ===
def _safe_float(x):
    try:
        x = float(x)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return x


# === Load all uploaded CSVs ===
def _load_all_csvs():
    data_dir = Path("data/uploads")
    dfs = []
    for p in data_dir.glob("*.csv"):
        df = pd.read_csv(p)
        if "Symbol" not in df.columns:
            df["Symbol"] = p.stem.upper()
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No CSVs found in data/uploads. Upload first.")
    return pd.concat(dfs, ignore_index=True)


# === Load raw history for a single symbol (from uploads) ===
def _load_history_for_symbol(symbol: str) -> pd.DataFrame:
    symbol = symbol.upper()
    data_dir = Path("data/uploads")
    # try exact filename first
    p = data_dir / f"{symbol}.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "Symbol" not in df.columns:
            df["Symbol"] = symbol
        return df
    # fallback: scan all files and filter by Symbol column if present
    dfs = []
    for q in data_dir.glob("*.csv"):
        df = pd.read_csv(q)
        if "Symbol" in df.columns:
            part = df[df["Symbol"].astype(str).str.upper() == symbol]
            if len(part):
                dfs.append(part)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    raise FileNotFoundError(f"No OHLCV history found for symbol {symbol} under data/uploads/.")


# === Feature Engineering ===
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
        g["atr"] = (g["High"] - g["Low"]).rolling(7).mean()
        g["volchg"] = g["Volume"].pct_change()
        g["ret5"] = g["Close"].pct_change(5)
        g["ret10"] = g["Close"].pct_change(10)
        g["ma_spread_7_14"] = (g["ma7"] - g["ma14"]) / g["ma14"]
        g["target"] = (g["Close"].shift(-1) > g["Close"]).astype(int)
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=FEATURES + ["target"], inplace=True)
    return df


# === Feature engineering for a single symbol dataframe (no Symbol col required) ===
def _engineer_single_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    df["Symbol"] = symbol
    return _engineer(df)


# === Save Feature Importance Chart ===
def _save_feature_importance(clf: LGBMClassifier, features: list[str], symbol: str):
    importances = clf.feature_importances_
    if importances is None or len(importances) == 0:
        return

    idx = np.argsort(importances)[::-1]
    top_k = min(15, len(idx))
    idx = idx[:top_k]

    plt.figure(figsize=(8, 6))
    plt.barh(range(top_k), importances[idx][::-1])
    plt.yticks(range(top_k), [features[i] for i in idx][::-1])
    plt.xlabel("Importance (gain)")
    plt.title(f"LightGBM Feature Importance – {symbol}")
    plt.tight_layout()

    out_path = ART_FI / f"feature_importance_{symbol}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[{symbol}] saved feature importance chart → {out_path}")


# === Train Per Symbol: Classifier + Regressor ===
def _train_per_symbol(df: pd.DataFrame):
    symbols = sorted(df["Symbol"].unique())
    results = {}
    for s in symbols:
        d = df[df["Symbol"] == s].copy()
        X = d[FEATURES]
        var = X.var(numeric_only=True)
        keep_feats = var[var > 1e-12].index.tolist() or FEATURES

        # build y targets before dropping
        d["y_reg"] = d["Close"].shift(-1)  # next-day Close

        d = d.dropna(subset=keep_feats + ["target", "Close", "y_reg"])

        y_class = d["target"].astype(int)
        y_reg = d["y_reg"]

        split = int(len(d) * 0.7)
        Xtr, Xte = d[keep_feats].iloc[:split], d[keep_feats].iloc[split:]
        ytr_cls, yte_cls = y_class.iloc[:split], y_class.iloc[split:]
        ytr_reg, yte_reg = y_reg.iloc[:split], y_reg.iloc[split:]

        if len(Xte) == 0:
            continue

        # === CLASSIFIER ===
        clf = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=1,
            random_state=42,
        )
        clf.fit(Xtr, ytr_cls)
        _save_feature_importance(clf, keep_feats, s)

        yp = clf.predict(Xte)
        prob = clf.predict_proba(Xte)[:, 1]
        acc = float(accuracy_score(yte_cls, yp)) if len(yte_cls.unique()) > 1 else 0.5
        try:
            auc = float(roc_auc_score(yte_cls, prob)) if len(yte_cls.unique()) > 1 else 0.5
        except Exception:
            auc = 0.5

        # === REGRESSOR ===
        reg = LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=1,
            random_state=42,
        )
        reg.fit(Xtr, ytr_reg)
        ypred_reg = reg.predict(Xte)

        # === SAVE ARTIFACTS ===
        joblib.dump(clf, ART / f"model_{s}.pkl")
        joblib.dump(reg, ART / f"model_reg_{s}.pkl")
        (ART / f"features_{s}.json").write_text(json.dumps(keep_feats))

        # === SAVE PREDICTIONS ===
        pd.DataFrame({
            "Date": d["Date"].iloc[split:].to_list(),
            **{f: Xte[f].to_list() for f in keep_feats},
            "p_up": prob,
            "pred": yp,
            "target": yte_cls.to_list(),
            "predicted_close": ypred_reg,
            "actual_close": yte_reg.to_list(),
        }).to_csv(ART / f"preds_{s}.csv", index=False)

        results[s] = {"acc": acc, "auc": auc, "features": keep_feats, "rows": int(len(d))}

    # === Rankings ===
    ranks = []
    for s in results:
        preds = pd.read_csv(ART / f"preds_{s}.csv")
        if len(preds) == 0:
            continue
        last = preds.iloc[-1]
        ranks.append({
            "symbol": s,
            "p_up": _safe_float(last["p_up"]),
            "acc": results[s]["acc"],
            "auc": results[s]["auc"],
            "predicted_close": _safe_float(last["predicted_close"]),
        })
    if ranks:
        pd.DataFrame(ranks).sort_values("p_up", ascending=False).to_csv(ART / "rankings.csv", index=False)
        ranking_file = str(ART / "rankings.csv")
    else:
        ranking_file = None

    return {"symbols": results, "ranking_file": ranking_file}


# === Bootstrap all ===
def bootstrap_all():
    raw = _load_all_csvs()
    df = _engineer(raw)
    info = _train_per_symbol(df)
    return {"status": "ok", **info}


# === INTERNAL: forecast forward to a target date (recursive daily) ===
def _forecast_to_date(symbol: str, target_date: pd.Timestamp):
    """
    Uses latest raw OHLCV for the symbol, recursively projects days until target_date.
    Returns dict with date/p_up/predicted_close for the target_date.
    """
    symbol = symbol.upper()
    # load models & keep_feats
    clf_path = ART / f"model_{symbol}.pkl"
    reg_path = ART / f"model_reg_{symbol}.pkl"
    feats_path = ART / f"features_{symbol}.json"
    if not (clf_path.exists() and reg_path.exists() and feats_path.exists()):
        return {"error": f"Missing model artifacts for {symbol}."}

    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    keep_feats = json.loads(feats_path.read_text())

    # load raw history for this symbol
    raw = _load_history_for_symbol(symbol)
    raw = raw.copy()
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values("Date").reset_index(drop=True)

    if len(raw) < 20:
        return {"error": f"Too few rows in raw history for {symbol}."}

    # engineer to get latest feature row
    eng = _engineer_single_symbol(raw[["Date","Open","High","Low","Close","Volume"]].copy(), symbol)
    if len(eng) == 0:
        return {"error": f"Not enough engineered rows for {symbol}."}

    last_known_date = eng["Date"].max()
    if target_date <= last_known_date:
        # shouldn’t happen here; caller handles <= case via stored preds
        target_date = last_known_date + pd.Timedelta(days=1)

    # we will append synthetic days to RAW (not eng), then re-engineer rolling features each loop
    current_raw = raw.copy()

    # carry forward last Volume to keep volchg computable
    last_vol = float(current_raw["Volume"].iloc[-1])

    # rolling forward day by day
    cur_date = last_known_date
    out_for_return = None

    while cur_date < target_date:
        # recompute engineered features to get the latest X row
        eng_loop = _engineer_single_symbol(
            current_raw[["Date","Open","High","Low","Close","Volume"]].copy(),
            symbol
        )
        if len(eng_loop) == 0:
            return {"error": f"Feature engineering failed during recursion for {symbol}."}

        # take the last row’s features as X
        X_last = eng_loop.iloc[[-1]][keep_feats]

        # predict next day close & up prob
        p_up = float(clf.predict_proba(X_last)[0, 1])
        next_close = float(reg.predict(X_last)[0])

        # synthesize next-day OHLCV
        prev_close = float(current_raw["Close"].iloc[-1])
        next_open = prev_close
        next_high = max(next_open, next_close)
        next_low = min(next_open, next_close)
        next_vol = last_vol  # carry forward

        next_date = cur_date + pd.Timedelta(days=1)

        # append synthetic row
        current_raw = pd.concat([
            current_raw,
            pd.DataFrame([{
                "Date": next_date,
                "Open": next_open,
                "High": next_high,
                "Low": next_low,
                "Close": next_close,
                "Volume": next_vol,
                "Symbol": symbol,
            }])
        ], ignore_index=True)

        cur_date = next_date
        out_for_return = {
            "date": str(next_date.date()),
            "p_up": _safe_float(p_up),
            "pred": 1 if p_up is not None and p_up >= 0.5 else 0,
            "predicted_close": _safe_float(next_close),
            "actual_close": None,
        }

    return out_for_return or {"error": "No forecast produced."}


# === Predict on specific date (historical or future) ===
def predict_on_date(date: str, symbol: str | None):
    out = {}
    qd = pd.to_datetime(date)
    for p in Path("artifacts").glob("preds_*.csv"):
        s = p.stem.replace("preds_", "")
        if symbol and s.upper() != symbol.upper():
            continue

        pdf = pd.read_csv(p)
        if "Date" not in pdf.columns:
            continue
        pdf["Date"] = pd.to_datetime(pdf["Date"])
        last_known = pdf["Date"].max()

        if qd <= last_known:
            # historical/stored
            pdf2 = pdf[pdf["Date"] <= qd]
            if len(pdf2) == 0:
                continue
            last = pdf2.iloc[-1]
            out[s] = {
                "date": str(last["Date"].date()),
                "p_up": _safe_float(last.get("p_up")),
                "pred": int(last.get("pred", 0)),
                "predicted_close": _safe_float(last.get("predicted_close")),
                "actual_close": _safe_float(last.get("actual_close")),
            }
        else:
            # future → forecast recursively from latest history
            fut = _forecast_to_date(s, qd)
            if isinstance(fut, dict) and "error" in fut:
                # if forecast failed, at least return latest known
                last = pdf.iloc[-1]
                out[s] = {
                    "date": str(qd.date()),
                    "p_up": None,
                    "pred": None,
                    "predicted_close": None,
                    "actual_close": None,
                    "note": f"Forecast failed: {fut['error']}. Returning empty fields.",
                    "last_known_snapshot": {
                        "date": str(last['Date'].date()),
                        "p_up": _safe_float(last.get("p_up")),
                        "predicted_close": _safe_float(last.get("predicted_close")),
                    }
                }
            else:
                out[s] = fut

    return {"query_date": date, "predictions": out}
