# backend/api.py
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List, Optional, Tuple

app = FastAPI(title="WillItRain API")

# Allow all origins (safe for hackathon/demo). Streamlit server-side requests don't need CORS,
# but if you ever call backend directly from browser, this avoids CORS blocking.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.joblib")  # relative to backend/
MODEL_URL = os.environ.get("MODEL_URL")  # optional: if you want backend to download model automatically
MODEL: Any = None
MODEL_LOADED: bool = False

class PredictRequest(BaseModel):
    # Either features (list) or raw aggregated fields
    features: Optional[List[float]] = None
    temp: Optional[float] = None
    hum: Optional[float] = None
    wind: Optional[float] = None
    precip: Optional[float] = None
    uv: Optional[float] = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED}

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

    # dictionary: iterate keys
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

def _download_model_if_url():
    """
    If MODEL_URL set and model file not present, try download to MODEL_PATH.
    (Simple implementation; Render environment should allow outbound HTTP.)
    """
    import requests
    if not MODEL_URL:
        return False
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

def load_model() -> bool:
    """
    Load model from MODEL_PATH.
    If the loaded object is a container (dict/list/obj), try to find estimator with 'predict'.
    Sets global MODEL and MODEL_LOADED. Returns True if an estimator was loaded.
    """
    global MODEL, MODEL_LOADED
    MODEL_LOADED = False
    MODEL = None

    # try to download if MODEL_PATH missing and MODEL_URL provided
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

    # if raw itself is estimator
    est, path = _find_estimator_in_obj(raw)
    if est is not None:
        MODEL = est
        MODEL_LOADED = True
        try:
            # add a tiny debug meta (not necessary but helpful in logs)
            setattr(MODEL, "_loaded_from_info", {"model_path": MODEL_PATH, "found_path": path, "raw_type": str(type(raw))})
        except Exception:
            pass
        print("Estimator found in model file. path:", "->".join(path) if path else "(root)")
        return True

    # nothing found: keep raw for inspection and report failure
    MODEL = raw
    MODEL_LOADED = False
    print("No estimator with 'predict' found inside loaded object. raw type:", type(raw))
    return False

@app.get("/model_info")
def model_info():
    """
    Diagnostic endpoint. Attempts to load model (if not loaded) and returns info about loaded object.
    """
    # attempt load if not yet loaded
    ok = load_model() if not MODEL_LOADED else True
    if MODEL_LOADED:
        info = {"status": "model_loaded", "estimator_type": str(type(MODEL))}
        try:
            meta = getattr(MODEL, "_loaded_from_info", {})
            info.update(meta)
        except Exception:
            pass
        return info
    else:
        # MODEL may be raw object or None
        return {"status": "no_model_loaded", "raw_type": str(type(MODEL)), "model_path": MODEL_PATH}

def features_from_request(req: PredictRequest) -> Optional[List[float]]:
    # prefer explicit features
    if req.features is not None:
        return list(req.features)
    # fallback to simple aggregated fields
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
    # ensure model loaded
    if not MODEL_LOADED:
        ok = load_model()
        if not ok:
            raise HTTPException(status_code=503, detail="Model not available on server.")
    feats = features_from_request(req)
    if feats is None:
        raise HTTPException(status_code=400, detail="No features supplied.")
    X = np.array([feats])
    try:
        # If MODEL is still raw (not estimator), this will raise and be returned as 500 with detail
        pred = MODEL.predict(X)
        probs = MODEL.predict_proba(X).tolist() if hasattr(MODEL, "predict_proba") else None
        return {"prediction": str(pred[0]), "probs": probs}
    except Exception as e:
        # include a bit more detail in logs for debugging
        print("Prediction failed:", repr(e))
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

# Try loading on startup (best-effort)
@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except Exception as e:
        print("Startup model load error:", e)
