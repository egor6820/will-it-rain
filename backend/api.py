# backend/api.py
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="WillItRain API")

MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.joblib")  # path relative to backend/
MODEL = None
MODEL_LOADED = False

class PredictRequest(BaseModel):
    # або передавайте список фіч:
    features: Optional[List[float]] = None
    # або "сирі" агреговані поля (фронтенд може відправити їх)
    temp: Optional[float] = None
    hum: Optional[float] = None
    wind: Optional[float] = None
    precip: Optional[float] = None
    uv: Optional[float] = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED}

def load_model():
    global MODEL, MODEL_LOADED
    if MODEL_LOADED:
        return True
    if not os.path.exists(MODEL_PATH):
        MODEL_LOADED = False
        return False
    try:
        # memory-map first (зменшує пік пам'яті)
        MODEL = joblib.load(MODEL_PATH, mmap_mode='r')
        MODEL_LOADED = True
        print("Model loaded with mmap:", MODEL_PATH)
        return True
    except Exception:
        try:
            MODEL = joblib.load(MODEL_PATH)
            MODEL_LOADED = True
            print("Model loaded:", MODEL_PATH)
            return True
        except Exception as e:
            print("Failed to load model:", e)
            MODEL_LOADED = False
            return False

def features_from_request(req: PredictRequest):
    if req.features is not None:
        return list(req.features)
    # мінімальний набір: [temp, hum, wind, precip, uv]
    if req.temp is None and req.hum is None:
        return None
    return [
        float(req.temp or 0.0),
        float(req.hum or 0.0),
        float(req.wind or 0.0),
        float(req.precip or 0.0),
        float(req.uv or 0.0),
    ]

@app.post("/predict")
def predict(req: PredictRequest):
    if not MODEL_LOADED:
        ok = load_model()
        if not ok:
            raise HTTPException(status_code=503, detail="Model not available on server.")
    feats = features_from_request(req)
    if feats is None:
        raise HTTPException(status_code=400, detail="No features supplied.")
    X = np.array([feats])
    try:
        pred = MODEL.predict(X)
        probs = MODEL.predict_proba(X).tolist() if hasattr(MODEL, "predict_proba") else None
        return {"prediction": str(pred[0]), "probs": probs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

# Попробуємо завантажити модель на старті (не обов'язково)
@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except Exception as e:
        print("Startup model load error:", e)
