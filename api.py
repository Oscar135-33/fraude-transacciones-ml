# api.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Fraud Detection API")

# --- Cargar artefactos ---
MODEL_PATH = "model_final.pkl"        
COLUMNS_PATH = "model_columns.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Modelo no encontrado en {MODEL_PATH}. Entrena el modelo y guarda el artefacto.")

model = joblib.load(MODEL_PATH)

scaler = None

if not os.path.exists(COLUMNS_PATH):
    raise RuntimeError(f"Lista de columnas no encontrada en {COLUMNS_PATH}. Guarda las columnas usadas en el modelo.")
model_columns = joblib.load(COLUMNS_PATH)  # lista en el orden que esperó el modelo

# --- Pydantic model flexible: recibimos un dict con feature->valor ---
class Transaction(BaseModel):
    features: Dict[str, float]

# --- Helper de preprocesado ---
def _prepare_input(features: Dict[str, Any], fill_missing_with_zero: bool = True) -> pd.DataFrame:
    """
    Construye un DataFrame de 1 fila con el orden de columnas esperado por el modelo.
    Si falta alguna columna, la rellena con 0.0 (o lanza error si fill_missing_with_zero=False).
    """
    row = {}
    for col in model_columns:
        if col in features:
            row[col] = features[col]
        else:
            if fill_missing_with_zero:
                row[col] = 0.0
            else:
                raise ValueError(f"Falta columna requerida: {col}")
    df = pd.DataFrame([row], columns=model_columns)

    # Aplicar escalado a Time/Amount si existe scaler
    # Se asume que el scaler fue entrenado para ['Time', 'Amount'] o compatibilidad similar.
    if scaler is not None:
        cols_to_scale = []
        for c in ['Time', 'Amount']:
            if c in df.columns:
                cols_to_scale.append(c)
        if cols_to_scale:
            try:
                df[cols_to_scale] = scaler.transform(df[cols_to_scale])
            except Exception as e:
                # Si el scaler fue guardado de otra forma, intenta aplicar columna por columna
                try:
                    df['Time'] = scaler.transform(df[['Time']])
                    df['Amount'] = scaler.transform(df[['Amount']])
                except Exception:
                    # No escalamos si falla; log minimal y seguimos
                    print("⚠️ No se pudo aplicar scaler automáticamente:", str(e))
    return df

# --- Endpoints ---
@app.get("/")
def root():
    return {"service": "fraud-detection-api", "status": "ok"}

@app.post("/predict")
def predict(txn: Transaction, threshold: Optional[float] = Query(0.5, ge=0.0, le=1.0), fill_missing_with_zero: bool = True):
    """
    Recibe JSON:
    {
      "features": {"V1": -1.23, "V2": 0.5, ..., "Amount": 100.0, "Time": 34567.0 }
    }
    Query params:
      - threshold: umbral para decidir is_fraud (default 0.5)
      - fill_missing_with_zero: si True rellena columnas faltantes con 0.0
    """
    try:
        X = _prepare_input(txn.features, fill_missing_with_zero=fill_missing_with_zero)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Predecir probabilidad
    try:
        proba = model.predict_proba(X)[:, 1][0]
    except Exception as e:
        # Algunos modelos (entornos) usan predict_proba diferente; intentamos fallback a decision_function
        try:
            score = model.decision_function(X)[0]
            proba = 1.0 / (1.0 + np.exp(-score))
        except Exception:
            raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

    is_fraud = int(proba >= threshold)
    return {"fraud_probability": float(proba), "is_fraud": is_fraud, "threshold": float(threshold)}

# --- Ejecutar con: uvicorn api:app --reload --host 0.0.0.0 --port 8000 ---