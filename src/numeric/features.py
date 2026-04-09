"""Feature engineering for the route-feasibility dataset."""

from __future__ import annotations

import pandas as pd

NUMERIC_COLS = ["range_km", "distance_km", "range_margin_ratio", "payload_proxy"]
BOOL_COLS = ["twin_engine", "etops_capable", "long_haul", "transoceanic"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["range_margin_ratio"] = out["distance_km"] / out["range_km"]
    out["long_haul"] = out["distance_km"] > 5000
    out["transoceanic"] = out["distance_km"] > 5500
    return out


def feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = add_features(df)
    if "payload_proxy" not in df.columns:
        df = df.assign(payload_proxy=0.5)
    X = df[NUMERIC_COLS + BOOL_COLS].astype(float)
    # add manufacturer one-hot if available
    if "manufacturer" in df.columns:
        man = pd.get_dummies(df["manufacturer"].fillna("unknown"), prefix="man")
        X = pd.concat([X, man], axis=1)
    y = df["feasible"].astype(int)
    return X, y
