from flask import Blueprint, request, jsonify
from pathlib import Path
import pandas as pd

try:
    from .hospital_recommendation import recommend_hospitals
except ImportError:
    from hospital_recommendation import recommend_hospitals

recommendation_bp = Blueprint(
    "hospital_recommendation",
    __name__
)

ROOT = Path(__file__).resolve().parent.parent

HF_REPO_ID = "sumera09/evidencebridge-karnataka-ml"
HF_DATA_FILENAME = "data/hospital_recommendation_validated.csv"

DATA_PATH = (
    ROOT / "data/processed/hospital_recommendation_validated.csv"
)

_hospitals = None

def load_hospitals():
    global _hospitals

    if _hospitals is None:
        if DATA_PATH.exists():
            dataset_path = DATA_PATH
        else:
            from huggingface_hub import hf_hub_download

            dataset_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_DATA_FILENAME,
                repo_type="model"
            )

        _hospitals = pd.read_csv(dataset_path)

    return _hospitals


@recommendation_bp.route(
    "/api/hospitals/recommend",
    methods=["POST"]
)
def hospital_recommendations():
    try:
        data = request.get_json(silent=True) or {}

        district = data.get("district")
        specialty = data.get("specialty")

        limit = data.get("limit", 10)
        require_coordinates = data.get(
            "require_coordinates", False
        )

        if not isinstance(limit, int) or isinstance(limit, bool):
            return jsonify({
                "success": False,
                "error": "limit must be an integer"
            }), 400

        if limit < 1 or limit > 50:
            return jsonify({
                "success": False,
                "error": "limit must be between 1 and 50"
            }), 400

        if not isinstance(require_coordinates, bool):
            return jsonify({
                "success": False,
                "error": "require_coordinates must be true or false"
            }), 400

        if district is not None and not isinstance(district, str):
            return jsonify({
                "success": False,
                "error": "district must be a string"
            }), 400

        if specialty is not None and not isinstance(specialty, str):
            return jsonify({
                "success": False,
                "error": "specialty must be a string"
            }), 400

        hospitals = load_hospitals()

        result = recommend_hospitals(
            hospitals,
            district=district,
            specialty=specialty,
            limit=limit,
            require_coordinates=require_coordinates
        )

        records = result.astype(object).where(
            pd.notna(result), None
        ).to_dict(orient="records")

        return jsonify({
            "success": True,
            "count": len(records),
            "filters": {
                "district": district,
                "specialty": specialty,
                "require_coordinates": require_coordinates
            },
            "data": records,
            "warning": (
                "These are directory-based candidates, not "
                "clinically validated hospital rankings. "
                "Independent verification is required."
            )
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Recommendation failed: " + str(e)
        }), 500


@recommendation_bp.route(
    "/api/hospitals/recommend/health",
    methods=["GET"]
)
def recommendation_health():
    try:
        hospitals = load_hospitals()

        return jsonify({
            "success": True,
            "service": "Hospital Recommendation",
            "status": "loaded",
            "hospital_records": len(hospitals),
            "districts": int(hospitals["District"].nunique()),
            "basis": "Directory filters and evidence completeness",
            "clinical_quality_assessed": False
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "status": "error",
            "error": str(e)
        }), 500
