import joblib
from fastapi import APIRouter
from app.core.model_loader import load_models
from sklearn.metrics import precision_score, recall_score, f1_score
from functools import lru_cache
import logging

@lru_cache()
def load_test_data():
    X_test = joblib.load("data/X_test.pkl")
    y_test = joblib.load("data/y_test.pkl")
    return X_test, y_test

logger = logging.getLogger("uvicorn")
router = APIRouter(prefix="/metrics", tags=["Metricas de las predicciones"])

models = load_models()
X_test, y_test = load_test_data()

precomputed_metrics = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    precomputed_metrics[name] = {
        "precision": round(precision_score(y_test, y_pred), 3),
        "recall": round(recall_score(y_test, y_pred), 3),
        "f1": round(f1_score(y_test, y_pred), 3)
    }

logger.info(f"Métricas precalculadas: {precomputed_metrics}")

@router.get("")
def metrics():
    return precomputed_metrics
