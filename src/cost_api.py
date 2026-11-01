
from flask import Blueprint, request, jsonify

try:
    from .predict_treatment_cost import predict_treatment_cost
except ImportError:
    from predict_treatment_cost import predict_treatment_cost

cost_bp = Blueprint("cost_prediction", __name__)

@cost_bp.route("/api/cost/predict", methods=["POST"])
def predict_cost():
    try:
        data = request.get_json(silent=True) or {}

        package_name = data.get("package_name")
        room_category = data.get("room_category")

        if not package_name or not room_category:
            return jsonify({
                "success": False,
                "error": "package_name and room_category are required"
            }), 400

        result = predict_treatment_cost(
            package_name,
            room_category
        )

        return jsonify({
            "success": True,
            "data": result
        }), 200

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Prediction service error: " + str(e)
        }), 500


@cost_bp.route("/api/cost/health", methods=["GET"])
def cost_health():
    return jsonify({
        "success": True,
        "service": "Treatment Package Cost Prediction",
        "model": "RandomForestRegressor",
        "status": "loaded",
        "scope": "Sakhii Hospital experimental dataset",
        "karnataka_wide_prediction": False
    }), 200
