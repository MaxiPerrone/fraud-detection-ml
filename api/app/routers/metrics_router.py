import logging
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import APIRouter
from sklearn.metrics import precision_score, recall_score, f1_score

from app.core.model_loader import load_models, load_scaler

logger = logging.getLogger("uvicorn")

router = APIRouter(
    prefix="/metrics",
    tags=["Metricas de las predicciones"]
)

models = {}
precomputed_metrics = {}

@lru_cache()
def load_test_data():
    base = Path(__file__).resolve().parents[2]
    X_path = base / "data" / "X_test.pkl"
    y_path = base / "data" / "y_test.pkl"
    X_test = joblib.load(X_path)
    y_test = joblib.load(y_path)
    return X_test, y_test


def compute_metrics():
    global precomputed_metrics

    X_test, y_test = load_test_data()
    scaler = load_scaler()

    # Muestra para que el startup no tarde años
    sample_size = min(4000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    y_sample = y_test.loc[X_sample.index]

    X_sample_scaled = scaler.transform(X_sample)

    precomputed_metrics = {}

    for name, model in models.items():
        try:
            y_pred = model.predict(X_sample_scaled)
            precomputed_metrics[name] = {
                "precision": round(precision_score(y_sample, y_pred), 3),
                "recall": round(recall_score(y_sample, y_pred), 3),
                "f1": round(f1_score(y_sample, y_pred), 3),
            }
        except Exception as e:
            logger.error(f"Error calculando métricas para {name}: {e}")

    logger.info(f"Métricas precalculadas: {precomputed_metrics}")


@router.on_event("startup")
def startup_event():
    global models
    models = load_models()
    compute_metrics()


@router.get("")
def metrics():
    return precomputed_metrics
