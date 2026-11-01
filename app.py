
from flask import Flask, jsonify
from src.cost_api import cost_bp

def create_app():
    app = Flask(__name__)

    # Register treatment package cost API
    app.register_blueprint(cost_bp)

    @app.route("/", methods=["GET"])
    def home():
        return jsonify({
            "project": "EvidenceBridge Karnataka ML API",
            "status": "running",
            "components": [
                "Treatment Package Cost Prediction"
            ]
        })

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "success": True,
            "service": "EvidenceBridge ML API",
            "status": "healthy"
        })

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
