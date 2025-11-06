from fastapi import FastAPI
from app.routers import predict_router, compare_router, metrics_router

app = FastAPI(
    title="Fraud Detection API",
    description="API REST para detección de fraude con modelos ML: Logistic Regression, Random Forest y SVM.",
    version="1.0.0"
)

app.include_router(predict_router.router)
app.include_router(compare_router.router)
app.include_router(metrics_router.router)

@app.get("/")
def root():
    return {"message": "API fraud"}