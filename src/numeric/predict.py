"""Inference helper — loads a trained numeric model and predicts feasibility."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from src.numeric.features import feature_matrix

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "numeric"


def load(name: str = "logreg"):
    # Note: xgboost segfaults on some Intel-mac builds at inference time.
    # logreg is the safe default; metrics show logreg matches xgboost within ~0.3 ROC-AUC.
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        # fall back to whichever model exists
        for fb in ("xgboost", "mlp", "logreg"):
            if (MODELS_DIR / f"{fb}.pkl").exists():
                path = MODELS_DIR / f"{fb}.pkl"
                break
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_one(variant: str, manufacturer: str, range_km: float,
                twin_engine: bool, etops_capable: bool,
                distance_km: float, model_name: str = "logreg") -> dict:
    bundle = load(model_name)
    model, feat_cols = bundle["model"], bundle["feature_columns"]
    row = pd.DataFrame([{
        "variant": variant,
        "manufacturer": manufacturer,
        "range_km": range_km,
        "twin_engine": twin_engine,
        "etops_capable": etops_capable,
        "distance_km": distance_km,
        "feasible": 0,  # dummy
    }])
    X, _ = feature_matrix(row)
    # align columns
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feat_cols]
    proba = float(model.predict_proba(X)[0, 1])
    return {"feasible": proba >= 0.5, "probability": proba}
