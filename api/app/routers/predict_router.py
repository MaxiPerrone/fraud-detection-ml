import numpy as np

from fastapi import APIRouter
from app.schemas.transaction import Transaction
from app.core.model_loader import load_models

router = APIRouter(prefix="/predict", tags=["Predicciones"])
models = load_models()

@router.post("/{model_name}")
def predict(model_name: str, txn: Transaction):
    if model_name not in models:
        return {"error": f"Model '{model_name}' not found. [logistic_regression, random_forest, svm]"}
    
    X = np.array([[
        txn.distance_from_home, 
        txn.distance_from_last_transaction, 
        txn.ratio_to_median_purchase_price, 
        txn.repeat_retailer, 
        txn.used_chip,
        txn.used_pin_number,
        txn.online_order]])
    
    model = models[model_name]
    y_pred = model.predict(X)[0]
    
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0][1]
        probability = round(float(prob), 3)
    else:
        probability = None

    return {
        'model': model_name,
        'prediction': 'Fraud' if y_pred == 1 else 'Legitime',
        'probability': probability
    }