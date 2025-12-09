import pandas as pd
from fastapi import APIRouter

from app.core.model_loader import load_models, load_scaler
from app.core.ensemble import ensemble_predict
from app.schemas.transaction import Transaction

router = APIRouter(prefix="/compare", tags=["Comparar modelos"])

models = load_models()
scaler = load_scaler()

@router.post("/")
def compare(txn: Transaction):

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

    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_scaled_df)[0]
        prob = (
            float(model.predict_proba(X_scaled_df)[0][1])
            if hasattr(model, "predict_proba")
            else None
        )

        results[name] = {
            "prediction": "Fraud" if y_pred == 1 else "Legit",
            "probability": round(prob, 3) if prob is not None else None
        }

    consensus = ensemble_predict(models, X_scaled_df)
    consensus_label = "Fraud" if consensus == 1 else "Legit"

    return {
        "results": results,
        "consensus": consensus_label
    }
