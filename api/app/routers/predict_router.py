from typing import Literal
import pandas as pd
from fastapi import APIRouter

from app.schemas.transaction import Transaction
from app.core.model_loader import load_models, load_scaler

router = APIRouter(prefix="/predict", tags=["Predicciones"])

models = load_models()
scaler = load_scaler()


@router.post("/{model_name}")
def predict(
    model_name: Literal["logistic_regression", "random_forest", "svm"],
    txn: Transaction
):
    X_raw = pd.DataFrame([{
        "distance_from_home": txn.distance_from_home,
        "distance_from_last_transaction": txn.distance_from_last_transaction,
        "ratio_to_median_purchase_price": txn.ratio_to_median_purchase_price,
        "repeat_retailer": txn.repeat_retailer,
        "used_chip": txn.used_chip,
        "used_pin_number": txn.used_pin_number,
        "online_order": txn.online_order,
    }])

    X_scaled = scaler.transform(X_raw)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X_raw.columns)

    model = models[model_name]
    y_pred = model.predict(X_scaled_df)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X_scaled_df)[0][1])

    return {
        "model": model_name,
        "prediction": "Fraud" if y_pred == 1 else "Legit",
        "probability": round(probability, 3) if probability is not None else None
    }
