# backend/api.py
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List, Optional, Tuple, Dict

app = FastAPI(title="WillItRain API")

# CORS: дозволяємо все для демо / хакатону (у проді звузити)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфіг
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.joblib")  # path relative to backend/
MODEL_URL = os.environ.get("MODEL_URL")  # optional: backend can download model if provided
MODEL: Any = None
MODEL_LOADED: bool = False

# --- Request schema
class PredictRequest(BaseModel):
    # або передавайте список фіч:
    features: Optional[List[float]] = None
    # або "сирі" агреговані поля (фронтенд може відправити їх)
    temp: Optional[float] = None
    hum: Optional[float] = None
    wind: Optional[float] = None
    precip: Optional[float] = None
    uv: Optional[float] = None

# --- Health
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED}

# --- Utility: search estimator inside arbitrary object
def _find_estimator_in_obj(obj: Any) -> Tuple[Optional[Any], Optional[List[str]]]:
    """
    Recursively search `obj` for the first sub-object that has a callable `.predict`.
    Returns (estimator, path_list) where path_list describes how we reached it (keys/indices).
    """
    try:
        if hasattr(obj, "predict") and callable(getattr(obj, "predict")):
            return obj, []
    except Exception:
        pass

    # dict: iterate keys
    if isinstance(obj, dict):
        for k, v in obj.items():
            est, path = _find_estimator_in_obj(v)
            if est is not None:
                return est, [str(k)] + (path or [])

    # list/tuple: iterate indices
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            est, path = _find_estimator_in_obj(v)
            if est is not None:
                return est, [str(i)] + (path or [])

    # generic object: try attributes
    try:
        attrs = getattr(obj, "__dict__", None)
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                est, path = _find_estimator_in_obj(v)
                if est is not None:
                    return est, [str(k)] + (path or [])
    except Exception:
        pass

    return None, None

# --- Optional: download model if MODEL_URL provided
def _download_model_if_url() -> bool:
    """
    If MODEL_URL set and model file not present, try download to MODEL_PATH.
    Returns True on success.
    """
    if not MODEL_URL:
        return False
    import requests
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    try:
        with requests.get(MODEL_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
        print("Downloaded model from MODEL_URL to", MODEL_PATH)
        return True
    except Exception as e:
        print("Failed to download model from MODEL_URL:", e)
        return False

# --- Load model robustly
def load_model() -> bool:
    """
    Load model from MODEL_PATH.
    If the loaded object is a container (dict/list/obj), try to find estimator with 'predict'.
    Sets global MODEL and MODEL_LOADED. Returns True if an estimator was loaded.
    """
    global MODEL, MODEL_LOADED
    MODEL_LOADED = False
    MODEL = None

    # try download if missing
    if not os.path.exists(MODEL_PATH) and MODEL_URL:
        _download_model_if_url()

    if not os.path.exists(MODEL_PATH):
        print("Model file not found at:", MODEL_PATH)
        return False

    raw = None
    # try mmap first (may fail for some objects)
    try:
        raw = joblib.load(MODEL_PATH, mmap_mode="r")
    except Exception as e_mmap:
        print("joblib.load with mmap failed:", e_mmap, "; retrying without mmap...")
        try:
            raw = joblib.load(MODEL_PATH)
        except Exception as e2:
            print("joblib.load failed:", e2)
            return False

    # if raw itself is estimator or contains estimator
    est, path = _find_estimator_in_obj(raw)
    if est is not None:
        MODEL = est
        MODEL_LOADED = True
        try:
            setattr(MODEL, "_loaded_from_info", {"model_path": MODEL_PATH, "found_path": path, "raw_type": str(type(raw))})
        except Exception:
            pass
        print("Estimator found in model file. path:", "->".join(path) if path else "(root)")
        return True

    # nothing found - keep raw for inspection and report failure
    MODEL = raw
    MODEL_LOADED = False
    print("No estimator with 'predict' found inside loaded object. raw type:", type(raw))
    return False

# --- Diagnostic endpoint
@app.get("/model_info")
def model_info():
    """
    Diagnostic endpoint. Attempts to load model (if not loaded) and returns info about loaded object.
    """
    ok = load_model() if not MODEL_LOADED else True
    if MODEL_LOADED:
        info: Dict[str, Any] = {"status": "model_loaded", "estimator_type": str(type(MODEL))}
        try:
            meta = getattr(MODEL, "_loaded_from_info", {})
            info.update(meta)
        except Exception:
            pass
        return info
    else:
        return {"status": "no_model_loaded", "raw_type": str(type(MODEL)), "model_path": MODEL_PATH}

# --- Features extraction from request
def features_from_request(req: PredictRequest) -> Optional[List[float]]:
    # prefer explicit features
    if req.features is not None:
        return list(req.features)
    # fallback to simple aggregated fields in order [temp, hum, wind, precip, uv]
    if req.temp is None and req.hum is None:
        return None
    return [
        float(req.temp or 0.0),
        float(req.hum or 0.0),
        float(req.wind or 0.0),
        float(req.precip or 0.0),
        float(req.uv or 0.0),
    ]

# --- Adapt features to model expectations
def _adapt_features_for_model(feats: List[float], model: Any) -> Tuple[List[float], str]:
    """
    Adjust the input features list to match model's expected number of features.
    Returns (new_feats, note).
    Rules:
      - If model has attribute n_features_in_ (sklearn) => use it.
      - If len(feats) == n_expected -> return as-is.
      - If len(feats) > n_expected -> trim from right (keep first n_expected).
      - If len(feats) < n_expected -> pad with zeros to the right.
    Note: This is a pragmatic temporary approach for demo; correct approach is mapping by feature names.
    """
    note = ""
    try:
        n_expected = getattr(model, "n_features_in_", None)
    except Exception:
        n_expected = None

    if n_expected is None:
        return feats, "model_n_features_unknown_no_change"

    try:
        n_expected = int(n_expected)
    except Exception:
        return feats, "model_n_features_not_int"

    if len(feats) == n_expected:
        return feats, "ok_same_length"
    elif len(feats) > n_expected:
        new = feats[:n_expected]
        note = f"trimmed_from_{len(feats)}_to_{n_expected}"
        return new, note
    else:
        padding = [0.0] * (n_expected - len(feats))
        new = feats + padding
        note = f"padded_from_{len(feats)}_to_{n_expected}"
        return new, note

# --- Predict endpoint
@app.post("/predict")
def predict(req: PredictRequest):
    # ensure model loaded
    if not MODEL_LOADED:
        ok = load_model()
        if not ok:
            raise HTTPException(status_code=503, detail="Model not available on server.")
    feats = features_from_request(req)
    if feats is None:
        raise HTTPException(status_code=400, detail="No features supplied.")

    # adapt features for the model (trim/pad) and report note if changed
    try:
        adapted_feats, note = _adapt_features_for_model(feats, MODEL)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature adaptation failed: {e}")

    X = np.array([adapted_feats])
    try:
        pred = MODEL.predict(X)
        probs = MODEL.predict_proba(X).tolist() if hasattr(MODEL, "predict_proba") else None
        resp: Dict[str, Any] = {"prediction": str(pred[0]), "probs": probs}
        if note and note != "ok_same_length":
            resp["note"] = note
            resp["original_features"] = feats
            resp["used_features"] = adapted_feats
        return resp
    except Exception as e:
        print("Prediction failed:", repr(e))
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

# --- Try loading on startup
@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except Exception as e:
        print("Startup model load error:", e)
