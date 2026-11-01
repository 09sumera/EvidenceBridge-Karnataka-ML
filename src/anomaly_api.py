from flask import Blueprint, request, jsonify
from pathlib import Path
from huggingface_hub import hf_hub_download
import pandas as pd
import joblib
import json

anomaly_bp = Blueprint("anomaly_detection", __name__)

REPO_ID = "sumera09/evidencebridge-karnataka-ml"

def load_anomaly_model():
    filenames = [
        "hospital_anomaly_isolation_forest.joblib",
        "hospital_anomaly_feature_columns.json",
        "hospital_anomaly_log_transform_columns.json",
        "hospital_anomaly_model_metadata.json"
    ]

    paths = {}

    for filename in filenames:
        paths[filename] = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"models/{filename}",
            repo_type="model"
        )

    model = joblib.load(paths[filenames[0]])

    with open(paths[filenames[1]]) as f:
        feature_columns = json.load(f)

    with open(paths[filenames[2]]) as f:
        log_columns = json.load(f)

    with open(paths[filenames[3]]) as f:
        metadata = json.load(f)

    return model, feature_columns, log_columns, metadata


_model, _feature_columns, _log_columns, _metadata = load_anomaly_model()


@anomaly_bp.route("/api/anomaly/predict", methods=["POST"])
def predict_anomaly():
    try:
        data = request.get_json(silent=True) or {}

        missing = [
            col for col in _feature_columns
            if col not in data
        ]

        if missing:
            return jsonify({
                "success": False,
                "error": "Missing required features",
                "missing_features": missing
            }), 400

        features = pd.DataFrame(
            [{col: data[col] for col in _feature_columns}]
        )

        for col in _log_columns:
            features[col] = pd.to_numeric(
                features[col], errors="coerce"
            ).clip(lower=0)

            features[col] = __import__("numpy").log1p(features[col])

        features = (
            features.apply(pd.to_numeric, errors="coerce")
            .replace([float("inf"), float("-inf")], float("nan"))
        )

        if features.isna().any().any():
            return jsonify({
                "success": False,
                "error": "Feature values must be valid numeric values"
            }), 400

        prediction = int(_model.predict(features)[0])
        score = float(-_model.decision_function(features)[0])

        return jsonify({
            "success": True,
            "data": {
                "anomaly_prediction": prediction,
                "anomaly_score": round(score, 6),
                "review_flag": (
                    "REVIEW" if prediction == -1 else "NOT_FLAGGED"
                ),
                "model": "IsolationForest",
                "model_purpose": "Hospital directory data-quality review",
                "records_used_for_training": _metadata["records_used"],
                "warning": (
                    "A review flag indicates an unusual directory record. "
                    "It is not proof of fraud or poor hospital quality."
                )
            }
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Anomaly prediction failed: " + str(e)
        }), 500


@anomaly_bp.route("/api/anomaly/health", methods=["GET"])
def anomaly_health():
    return jsonify({
        "success": True,
        "service": "Hospital Anomaly Detection",
        "model": "IsolationForest",
        "status": "loaded",
        "training_records": _metadata["records_used"],
        "feature_count": len(_feature_columns),
        "purpose": _metadata["purpose"],
        "limitations": _metadata["limitations"]
    }), 200
