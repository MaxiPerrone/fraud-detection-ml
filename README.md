# Fraud Detection API – Guía de Instalación y Ejecución

Este proyecto implementa una API REST para detección de fraude utilizando modelos de Machine Learning (Logistic Regression, Random Forest y SVM), entrenados sobre el dataset público Card Transaction Fraud Detection.

1. Librerías necesarias\
Instalar las dependencias principales:\
pip install fastapi uvicorn scikit-learn pandas numpy joblib kagglehub

2. Preparación del dataset\
El dataset se descarga automáticamente mediante kagglehub.\
Ejecutar primero desde /api/data:\
python load_features.py\
Este script genera dos archivos:
X_test.pkl
y_test.pkl\
Luego ejecutar:\
python train_models.py

3. Ejecutar la API\
Desde la carpeta raíz del backend:\
cd api\
python -m uvicorn app.server:app --reload

5. Acceder a Swagger:
http://127.0.0.1:8000/docs
