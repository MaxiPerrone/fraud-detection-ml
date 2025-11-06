import numpy as np
from fastapi import APIRouter
from app.core.model_loader import load_models
from app.core.ensemble import ensemble_predict
from app.schemas.transaction import Transaction

router = APIRouter(prefix="/compare", tags=["Comparar modelos"])
models = load_models()

@router.post("/")
def compare(txn: Transaction):
    X = np.array([[
        txn.distance_from_home, 
        txn.distance_from_last_transaction, 
        txn.ratio_to_median_purchase_price, 
        txn.repeat_retailer, 
        txn.used_chip,
        txn.used_pin_number,
        txn.online_order]])
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0, 1] if hasattr(model, "predict_proba") else None
        results[name] = {
            "prediction": "Fraud" if y_pred == 1 else "Legit",
            "probbability": round(float(prob), 3) if prob is not None else None
        }

    consensus = "Fraud" if ensemble_predict(models, X) == 1 else "Legit"

    return {
        "results": results, "consensus": consensus
    }