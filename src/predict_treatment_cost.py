import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Public Hugging Face repository
REPO_ID = "sumera09/evidencebridge-karnataka-ml"

# Download trained model
MODEL_PATH = hf_hub_download(
    repo_id=REPO_ID,
    filename="models/treatment_package_cost_model.joblib",
    repo_type="model"
)

# Download supporting package-price data
DATA_PATH = hf_hub_download(
    repo_id=REPO_ID,
    filename="data/COST003_Sakhii_package_prices.csv",
    repo_type="model"
)

# Load model and data
_model = joblib.load(MODEL_PATH)
_data = pd.read_csv(DATA_PATH)


def predict_treatment_cost(package_name, room_category):

    matches = _data[
        (_data["Procedure_Package"] == package_name) &
        (_data["Room_Category"] == room_category)
    ]

    if matches.empty:
        raise ValueError(
            "No published reference price for this package-room combination."
        )

    actual_price = float(matches.iloc[0]["Quoted_Price_INR"])

    features = pd.DataFrame([{
        "Procedure_Package": package_name,
        "Room_Category": room_category
    }])

    estimated_price = float(_model.predict(features)[0])

    return {
        "hospital": "Sakhii Hospital",
        "district": "Bengaluru",
        "package": package_name,
        "room_category": room_category,
        "published_reference_price_inr": round(actual_price, 2),
        "experimental_ml_estimate_inr": round(estimated_price, 2),
        "difference_inr": round(estimated_price - actual_price, 2),
        "price_type": "Published hospital package tariff",
        "model_status": "Experimental",
        "karnataka_wide_prediction": False,
        "package_scope_verified": False
    }
