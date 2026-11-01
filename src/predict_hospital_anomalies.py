
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def predict_hospital_anomalies(features, model_dir):
    """
    Predict unusual hospital directory records for manual data-quality review.

    features: DataFrame containing the required engineered features.
    model_dir: Directory containing the saved model and JSON configuration.

    Returns: DataFrame with anomaly_prediction, anomaly_score, review_flag.
    """
    model_dir = Path(model_dir)

    model = joblib.load(model_dir / "hospital_anomaly_isolation_forest.joblib")

    with open(model_dir / "hospital_anomaly_feature_columns.json") as f:
        feature_columns = json.load(f)

    with open(model_dir / "hospital_anomaly_log_transform_columns.json") as f:
        log_columns = json.load(f)

    missing = [c for c in feature_columns if c not in features.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = features[feature_columns].copy()

    for col in log_columns:
        X[col] = np.log1p(
            pd.to_numeric(X[col], errors="coerce").clip(lower=0)
        )

    X = (
        X.apply(pd.to_numeric, errors="coerce")
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0)
    )

    predictions = model.predict(X)
    scores = -model.decision_function(X)

    return pd.DataFrame({
        "anomaly_prediction": predictions,
        "anomaly_score": scores,
        "review_flag": np.where(
            predictions == -1, "REVIEW", "NOT_FLAGGED"
        )
    }, index=features.index)
