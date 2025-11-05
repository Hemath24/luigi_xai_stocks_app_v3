from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


def _load_artifacts(symbol: str):
    model = joblib.load(Path("artifacts") / f"model_{symbol}.pkl")
    feats = json.loads((Path("artifacts") / f"features_{symbol}.json").read_text())
    preds = pd.read_csv(Path("artifacts") / f"preds_{symbol}.csv")
    return model, feats, preds


def run_lime(symbol: str):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as e:
        return {"status": "error", "message": f"LIME not installed: {e}"}

    try:
        clf, feats, preds = _load_artifacts(symbol)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    if len(preds) == 0:
        return {"status": "error", "message": "No prediction rows available for LIME."}

    X = preds[feats].to_numpy()
    explainer = LimeTabularExplainer(
        X,
        feature_names=feats,
        class_names=[0, 1],
        discretize_continuous=True,
        random_state=42,
    )
    i = len(preds) - 1
    exp = explainer.explain_instance(
        X[i], clf.predict_proba, num_features=min(10, len(feats))
    )
    as_list = exp.as_list()
    outdir = Path("artifacts") / "xai" / symbol
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(as_list, columns=["feature", "weight"]).to_csv(
        outdir / "lime_last_row.csv", index=False
    )
    return {"status": "ok", "artifact": str(outdir / "lime_last_row.csv")}


def run_anchors(symbol: str):
    try:
        from alibi.explainers import AnchorTabular
        # Discretizer may or may not be compatible in your version; we'll try builtin first
        try:
            from alibi.utils.discretizer import Discretizer  # optional
            HAS_DISC = True
        except Exception:
            HAS_DISC = False
    except Exception as e:
        return {"status": "error", "message": f"Anchors not installed: {e}"}

    try:
        clf, feats, preds = _load_artifacts(symbol)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    if len(preds) == 0:
        return {"status": "error", "message": "No prediction rows available for Anchors."}

    # data
    X = preds[feats].to_numpy()
    i = len(preds) - 1

    # predict labels (0/1), as AnchorTabular expects discrete class output
    def _predict_labels(z):
        probs = clf.predict_proba(np.asarray(z))[:, 1]
        return (probs >= 0.5).astype(int)

    # === Strategy A: use AnchorTabular's built-in discretization (most portable) ===
    try:
        explainer = AnchorTabular(_predict_labels, feature_names=feats)
        # Many alibi versions accept these kwargs to discretize automatically:
        # disc_perc=True => percentile binning; disc_bins=8 => number of bins
        explainer.fit(X, disc_perc=True, disc_bins=8)
        exp = explainer.explain(X[i], threshold=0.85)
    except Exception as eA:
        # === Strategy B: try Discretizer with different signatures ===
        if not HAS_DISC:
            return {"status": "error", "message": f"Anchors fallback failed (no Discretizer available): {eA}"}

        try:
            # 1) Newer-ish signature
            disc = Discretizer(
                X,
                feature_names=feats,
                percentiles=True,
                n_bins=8,
            )
        except TypeError:
            try:
                # 2) Older-ish signature
                disc = Discretizer(
                    X,
                    feature_names=feats,
                    percentile_bins=True,
                    bins=8,
                )
            except Exception as eB:
                return {"status": "error", "message": f"Anchors discretizer incompatible: {eA} | {eB}"}

        try:
            disc.fit(X)
            cat_X = disc.discretize(X)

            # Some alibi versions take (predict_fn, discretizer) directly:
            try:
                explainer = AnchorTabular(_predict_labels, disc)
            except Exception:
                # Others take feature_names and use disc via fit arguments
                explainer = AnchorTabular(_predict_labels, feature_names=feats)

            # Try both fit styles
            try:
                explainer.fit(cat_X, disc.names)
                to_explain = cat_X[i]
            except Exception:
                explainer.fit(X, disc_perc=True, disc_bins=8)
                to_explain = X[i]

            exp = explainer.explain(to_explain, threshold=0.85)
        except Exception as eC:
            return {"status": "error", "message": f"Anchors failed after fallbacks: {eA} | {eC}"}

    outdir = Path("artifacts") / "xai" / symbol
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "anchors_last_row.json").write_text(json.dumps({
        "anchor": list(map(str, getattr(exp, "anchor", []))),
        "precision": float(getattr(exp, "precision", np.nan)),
        "coverage": float(getattr(exp, "coverage", np.nan)),
    }))
    return {"status": "ok", "artifact": str(outdir / "anchors_last_row.json")}



def run_dice(symbol: str, total_CFs: int = 3, desired_class: int = 1):
    try:
        import dice_ml
        from dice_ml import Dice
    except Exception as e:
        return {"status": "error", "message": f"DiCE not installed: {e}"}

    try:
        clf, feats, preds = _load_artifacts(symbol)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    if len(preds) == 0:
        return {"status": "error", "message": "No prediction rows available for DiCE."}

    X = preds[feats].reset_index(drop=True)
    y = (
        preds["target"].astype(int).reset_index(drop=True)
        if "target" in preds.columns
        else pd.Series(np.zeros(len(X), int))
    )

    backend = dice_ml.Model(model=clf, backend="sklearn", model_type="classifier")
    data = dice_ml.Data(
        dataframe=pd.concat([X, y.rename("target")], axis=1),
        continuous_features=feats,
        outcome_name="target",
    )
    exp = Dice(data, backend)

    query = X.iloc[[-1]].copy()
    try:
        dice = exp.generate_counterfactuals(
            query,
            total_CFs=total_CFs,
            desired_class=desired_class,
            features_to_vary=feats,
        )
        cf_df = dice.cf_examples_list[0].final_cfs_df
    except Exception as e:
        return {"status": "error", "message": f"DiCE generation failed: {e}"}

    outdir = Path("artifacts") / "xai" / symbol
    outdir.mkdir(parents=True, exist_ok=True)
    cf_df.to_csv(outdir / "dice_counterfactuals.csv", index=False)
    return {"status": "ok", "artifact": str(outdir / "dice_counterfactuals.csv")}


def run_ebm(symbol: str):
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except Exception as e:
        return {"status": "error", "message": f"interpret not installed: {e}"}

    try:
        clf, feats, preds = _load_artifacts(symbol)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    if "target" not in preds.columns:
        return {"status": "error", "message": "No target column in predictions CSV."}
    if len(preds) == 0:
        return {"status": "error", "message": "No prediction rows available for EBM."}

    X = preds[feats]
    y = preds["target"].astype(int)

    ebm = ExplainableBoostingClassifier(random_state=42)
    ebm.fit(X, y)

    # ---- FIX: use global explanation to get importances (no feature_importances_ in newer interpret) ----
    try:
        global_exp = ebm.explain_global()
        data = global_exp.data()
        names = data.get("names") or data.get("feature_names") or feats
        scores = data.get("scores") or data.get("overall_importance")
        if scores is None:
            # rare fallback: build a simple proxy via std of term scores
            scores = [abs(getattr(ebm, "term_importances_", [0]*len(names))[i]) for i in range(len(names))]
        imp = pd.DataFrame({"feature": names, "importance": scores}).sort_values(
            "importance", ascending=False
        )
    except Exception as e:
        return {"status": "error", "message": f"EBM importance extraction failed: {e}"}

    outdir = Path("artifacts") / "xai" / symbol
    outdir.mkdir(parents=True, exist_ok=True)
    imp_path = outdir / "ebm_importance.csv"
    imp.to_csv(imp_path, index=False)

    # Optional: effect curve for top feature (simple grid)
    try:
        topf = imp.iloc[0]["feature"]
        xs = np.linspace(X[topf].quantile(0.01), X[topf].quantile(0.99), 50)
        base = X.median(numeric_only=True).to_dict()
        grid = pd.DataFrame([{**base, topf: v} for v in xs])[feats]
        probs = ebm.predict_proba(grid)[:, 1]

        plt.figure(figsize=(5, 3))
        plt.plot(xs, probs)
        plt.title(f"EBM effect: {topf}")
        plt.xlabel(topf)
        plt.ylabel("P(up)")
        figp = outdir / "ebm_top_feature.png"
        plt.tight_layout()
        plt.savefig(figp, dpi=150)
        plt.close()
        artifacts = [str(imp_path), str(figp)]
    except Exception:
        # still return importance even if plot fails
        artifacts = [str(imp_path)]

    return {"status": "ok", "artifacts": artifacts}

