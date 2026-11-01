from flask import Blueprint, request, jsonify
import pandas as pd
import json

try:
    from .evidence_scoring import score_hospital_evidence, EVIDENCE_FIELDS
except ImportError:
    from evidence_scoring import score_hospital_evidence, EVIDENCE_FIELDS

evidence_bp = Blueprint("evidence_assessment", __name__)

REQUIRED_FIELDS = list(EVIDENCE_FIELDS.values())


@evidence_bp.route("/api/evidence/assess", methods=["POST"])
def assess_evidence():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "success": False,
                "error": "A JSON hospital record is required"
            }), 400

        missing = [field for field in REQUIRED_FIELDS if field not in data]

        if missing:
            return jsonify({
                "success": False,
                "error": "Missing required hospital fields",
                "missing_fields": missing
            }), 400

        hospital_df = pd.DataFrame([data])

        result = score_hospital_evidence(hospital_df)

        result_data = json.loads(
            result.to_json(orient="records")
        )[0]

        return jsonify({
            "success": True,
            "data": result_data,
            "assessment_scope": "Directory information completeness only"
        }), 200

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Evidence assessment failed: " + str(e)
        }), 500


@evidence_bp.route("/api/evidence/health", methods=["GET"])
def evidence_health():
    return jsonify({
        "success": True,
        "service": "Hospital Directory Evidence Assessment",
        "status": "loaded",
        "evidence_fields": len(REQUIRED_FIELDS),
        "score_range": "0-100",
        "limitations": [
            "Measures information availability only",
            "Does not verify information accuracy",
            "Does not establish registration or accreditation",
            "Does not measure clinical quality or patient safety"
        ]
    }), 200
