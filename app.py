# backend/api.py
"""
Robust FastAPI backend for "WillItRain" style predictor.
- Exposes /health, /info, /predict
- Loads model from MODEL_PATH (env)
- Adapts feature vector length when possible and returns clear diagnostics
"""

import os
import traceback
import joblib
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="WillItRain API", version="1.0")

# Allow all origins by default (adjust for security in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config: environment
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.joblib")
# Optional: comma-separated list of feature names in the order the frontend sends them.
# Example: "temp,hum,wind,precip,uv"
MODEL_FEATURES_ENV = os.environ.get("MODEL_FEATURES", None)

# Global model state
MODEL = None
MODEL_LOADED = False
MODEL_INFO: Dict[str, Any] = {}

class PredictRequest(BaseModel):
    # Either send `features` (ordered list) OR named values (temp/hum/...)
    features: Optional[List[float]] = None
    temp: Optional[float] = None
    hum: Optional[float] = None
    wind: Optional[float] = None
    precip: Optional[float] = None
    uv: Optional[float] = None

def debug_print(*args, **kwargs):
    # Print to stdout so Render logs capture it
    print(*args, **kwargs)

def load_model() -> bool:
    global MODEL, MODEL_LOADED, MODEL_INFO
    if MODEL_LOADED:
        return True
    debug_print(f"Trying to load model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        debug_print("Model file not found:", MODEL_PATH)
        MODEL_LOADED = False
        return False
    last_err = None
    try:
        MODEL = joblib.load(MODEL_PATH, mmap_mode='r')
        MODEL_LOADED = True
        debug_print("Model loaded (mmap_mode=r).")
    except Exception as e1:
        last_err = e1
        debug_print("joblib.load with mmap failed, retrying without mmap:", e1)
        try:
            MODEL = joblib.load(MODEL_PATH)
            MODEL_LOADED = True
            debug_print("Model loaded (no mmap).")
        except Exception as e2:
            debug_print("Failed to load model:", e2)
            debug_print(traceback.format_exc())
            MODEL_LOADED = False
            return False

    # Gather some info about model to return via /info
    MODEL_INFO = {
        "type": type(MODEL).__name__,
        "has_predict": hasattr(MODEL, "predict"),
        "has_predict_proba": hasattr(MODEL, "predict_proba"),
    }
    # if scikit-learn model with n_features_in_ or feature_names_in_:
    if hasattr(MODEL, "n_features_in_"):
        try:
            MODEL_INFO["n_features_in_"] = int(getattr(MODEL, "n_features_in_"))
        except Exception:
            pass
    if hasattr(MODEL, "feature_names_in_"):
        try:
            fn = getattr(MODEL, "feature_names_in_")
            MODEL_INFO["feature_names_in_"] = list(fn)
        except Exception:
            pass
    debug_print("Model info:", MODEL_INFO)
    return MODEL_LOADED

def features_from_request(req: PredictRequest) -> (Optional[List[float]], Dict[str, Any]):
    """
    Return (features_list_or_None, original_named_features_dict)
    original_named_features_dict useful for debug
    """
    orig = {}
    # if features list provided, trust its order (frontend must document order)
    if req.features is not None:
        try:
            feats = [float(x) for x in req.features]
            orig = {"features": feats}
            return feats, orig
        except Exception:
            return None, {}
    # otherwise build from known named fields (keep ordering consistent)
    named_order = ["temp", "hum", "wind", "precip", "uv"]
    provided = []
    for n in named_order:
        v = getattr(req, n)
        if v is not None:
            provided.append(float(v))
        else:
            # if any named field is absent, append None (we'll replace by 0.0 later)
            provided.append(None)
        orig[n] = getattr(req, n)
    # If all None -> no features
    if all(x is None for x in provided):
        return None, {}
    # Replace None by 0.0 (explicit) but record it
    provided_filled = [0.0 if x is None else float(x) for x in provided]
    return provided_filled, orig

def adapt_features_for_model(x: List[float]) -> (np.ndarray, List[str], Optional[str]):
    """
    Try to adapt input features list to model's expected input size.
    - If model provides feature_names_in_ -> attempt to map by names using MODEL_FEATURES_ENV or default order.
    - If mismatch, we either truncate (drop tail) or pad with zeros and include note.
    Returns: X_arr (2D), used_feature_names (order), note
    """
    note = None
    # ensure model loaded
    global MODEL
    # initial incoming features and default names
    default_names = ["temp", "hum", "wind", "precip", "uv"]
    incoming = list(x)
    incoming_names = []
    # if MODEL_FEATURES_ENV set, use that as incoming order
    if MODEL_FEATURES_ENV:
        env_names = [s.strip() for s in MODEL_FEATURES_ENV.split(",") if s.strip()]
        if len(env_names) != len(incoming):
            # if env specified different length, still accept env order for first min()
            incoming_names = env_names[:len(incoming)]
        else:
            incoming_names = env_names
    else:
        incoming_names = default_names[:len(incoming)]

    # figure out what model expects
    expected_n = None
    model_feature_names = None
    if hasattr(MODEL, "n_features_in_"):
        try:
            expected_n = int(getattr(MODEL, "n_features_in_"))
        except Exception:
            expected_n = None
    if hasattr(MODEL, "feature_names_in_"):
        try:
            model_feature_names = list(getattr(MODEL, "feature_names_in_"))
            # ensure expected_n consistent
            expected_n = expected_n or len(model_feature_names)
        except Exception:
            model_feature_names = None

    # If model doesn't give expected length, try to infer from coef_ or similar
    if expected_n is None:
        try:
            # many sklearn models expose coef_. We'll try to infer number of features
            coef = getattr(MODEL, "coef_", None)
            if coef is not None:
                expected_n = int(np.array(coef).shape[-1])
        except Exception:
            expected_n = None

    # Default: if still unknown, assume incoming size is fine
    if expected_n is None:
        expected_n = len(incoming)

    used_names = None
    adapted = list(incoming)

    if expected_n == len(incoming):
        used_names = incoming_names
        note = None
    elif expected_n < len(incoming):
        # truncate: choose first expected_n features
        adapted = adapted[:expected_n]
        used_names = incoming_names[:expected_n]
        note = f"Input had {len(incoming)} features; truncated to first {expected_n} features to match model which expects {expected_n} features."
    else:
        # expected_n > incoming; pad with zeros
        pad_len = expected_n - len(incoming)
        adapted = adapted + [0.0] * pad_len
        used_names = incoming_names + [f"pad_{i}" for i in range(pad_len)]
        note = f"Input had {len(incoming)} features; padded with {pad_len} zeros to match model which expects {expected_n} features."

    # If model provides feature_names, try to reorder/adapt by name (best-effort)
    if model_feature_names:
        # we'll attempt to map model_feature_names from incoming_names
        mapped_values = []
        mapped_used_names = []
        missing = []
        for mf in model_feature_names:
            if mf in incoming_names:
                idx = incoming_names.index(mf)
                mapped_values.append(incoming[idx])
                mapped_used_names.append(mf)
            else:
                # fallback: if we still have adapted values left, take next; else pad 0
                missing.append(mf)
                mapped_values.append(0.0)
                mapped_used_names.append(mf)
        adapted = mapped_values
        used_names = mapped_used_names
        note_extra = ""
        if missing:
            note_extra = f" Some model feature names were not present in input: {missing}. They were set to 0."
            note = (note or "") + note_extra

    X = np.array([adapted], dtype=float)
    return X, used_names, note

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED}

@app.get("/info")
def info():
    ok = load_model()
    return {"model_loaded": ok, "model_info": MODEL_INFO, "model_path": MODEL_PATH, "model_features_env": MODEL_FEATURES_ENV}

@app.post("/predict")
async def predict(req: PredictRequest, request: Request):
    """
    Accepts either:
      - {"features": [f1, f2, ...]}  (ordered list)
      - {"temp":..., "hum":..., "wind":..., "precip":..., "uv":...} (any subset)
    Returns structured json with prediction, probs (if any), used features and notes.
    """
    # ensure model is loaded
    if not MODEL_LOADED:
        ok = load_model()
        if not ok:
            raise HTTPException(status_code=503, detail="Model not available on server.")

    feats, orig = features_from_request(req)
    if feats is None:
        raise HTTPException(status_code=400, detail="No features supplied. Provide 'features' list or named fields (temp, hum, wind, precip, uv).")

    # Adapt features to model expectation
    try:
        X, used_feature_names, note = adapt_features_for_model(feats)
    except Exception as e:
        debug_print("Feature adaptation failed:", e)
        raise HTTPException(status_code=500, detail=f"Feature adaptation failed: {e}")

    # Call model.predict safely
    try:
        # Some saved objects might be plain dicts (rare). Guard against missing predict.
        if not hasattr(MODEL, "predict"):
            raise AttributeError("Loaded model object has no 'predict' method.")
        pred = MODEL.predict(X)
        # convert to python types
        if isinstance(pred, np.ndarray):
            pred_out = pred.tolist()
        else:
            pred_out = pred
        # probability (if available)
        probs = None
        try:
            if hasattr(MODEL, "predict_proba"):
                p = MODEL.predict_proba(X)
                probs = np.array(p).tolist()
        except Exception as e:
            # don't fail overall just because predict_proba broke
            debug_print("predict_proba failed:", e)
            probs = None

        # prepare model_info snapshot
        model_snapshot = MODEL_INFO.copy()
        # if model has n_features_in_, include it
        if hasattr(MODEL, "n_features_in_"):
            try:
                model_snapshot["n_features_in_"] = int(getattr(MODEL, "n_features_in_"))
            except Exception:
                pass

        response = {
            "prediction": pred_out[0] if isinstance(pred_out, list) and len(pred_out) == 1 else pred_out,
            "probs": probs,
            "original_features": orig,
            "used_features": used_feature_names,
            "model_info": model_snapshot,
            "note": note,
        }
        return response
    except Exception as e:
        debug_print("Model prediction failed:", e, traceback.format_exc())
        # Friendly error back to caller
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

# Try to load model on startup
@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except Exception as e:
        debug_print("Startup model load error:", e)
