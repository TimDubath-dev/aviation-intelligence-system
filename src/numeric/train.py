"""Train and compare three models on the route-feasibility dataset.

Models: Logistic Regression, MLP, XGBoost.
Metrics: accuracy, F1, ROC-AUC, Brier (calibration), 5-fold CV.
Hard-segment eval: performance on the *interesting* band where
distance / range ∈ [0.7, 1.1] — i.e. routes near each plane's limit, where
unobserved factors (headwind, payload) actually matter.
Outputs:
    models/numeric/{name}.pkl
    models/numeric/metrics.json
    models/numeric/calibration.png
    models/numeric/permutation_importance.json
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.numeric.features import feature_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "processed" / "route_dataset.csv"
MODELS_DIR = REPO_ROOT / "models" / "numeric"
RANDOM_STATE = 42


def make_models() -> dict:
    models = {
        "logreg": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
        ),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=300,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    except ImportError:
        print("xgboost not installed — skipping")
    return models


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
    }


def cross_val(model, X, y, k: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr, va in skf.split(X, y):
        m = pickle.loads(pickle.dumps(model))
        m.fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
    return {
        "cv_roc_auc_mean": float(np.mean(aucs)),
        "cv_roc_auc_std": float(np.std(aucs)),
    }


def hard_mask(X: pd.DataFrame) -> pd.Series:
    """Routes where distance/range is in the difficult band [0.7, 1.1]."""
    r = X["distance_km"] / X["range_km"]
    return (r >= 0.7) & (r <= 1.1)


def plot_calibration(models: dict, X_test, y_test) -> None:
    plt.figure(figsize=(6, 6))
    for name, m in models.items():
        proba = m.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=15, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration — route feasibility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "calibration.png", dpi=140)
    plt.close()


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    X, y = feature_matrix(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}  pos rate: {y.mean():.2%}")

    metrics: dict = {}
    fitted: dict = {}
    for name, model in make_models().items():
        print(f"\n=== {name} ===")
        model.fit(X_tr, y_tr)
        fitted[name] = model

        m_overall = evaluate(model, X_te, y_te)
        mask = hard_mask(X_te)
        m_hard = evaluate(model, X_te[mask], y_te[mask]) if mask.sum() > 50 else {}
        m_cv = cross_val(model, X_tr, y_tr)

        metrics[name] = {"overall": m_overall, "hard_segment": m_hard, **m_cv}
        print(json.dumps(metrics[name], indent=2))

        with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump({"model": model, "feature_columns": X.columns.tolist()}, f)

    # calibration plot
    plot_calibration(fitted, X_te, y_te)
    print(f"\nCalibration plot → {MODELS_DIR / 'calibration.png'}")

    # permutation importance on the best model (xgb if present, else logreg)
    best_name = "xgboost" if "xgboost" in fitted else "logreg"
    print(f"\nPermutation importance on {best_name} ...")
    perm = permutation_importance(
        fitted[best_name], X_te, y_te, n_repeats=5, random_state=RANDOM_STATE, n_jobs=1
    )
    importance = (
        pd.DataFrame(
            {"feature": X.columns, "mean": perm.importances_mean, "std": perm.importances_std}
        )
        .sort_values("mean", ascending=False)
        .head(15)
    )
    print(importance.to_string(index=False))
    importance.to_json(MODELS_DIR / "permutation_importance.json", orient="records", indent=2)

    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics → {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
