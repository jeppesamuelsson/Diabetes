import os, joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from config import FEATURE_NAMES

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/model.pkl")

class Features(BaseModel):
    age: float = Field(...)
    sex: float = Field(...)
    bmi: float = Field(...)
    bp: float = Field(...)
    s1: float = Field(...)
    s2: float = Field(...)
    s3: float = Field(...)
    s4: float = Field(...)
    s5: float = Field(...)
    s6: float = Field(...)

app = FastAPI(title="Diabetes Progression Risk Service")

def _load_model():
    try:
        obj = joblib.load(MODEL_PATH)
        return obj["pipeline"], obj.get("version","unknown")
    except Exception as e:
        raise RuntimeError(f"Model load failed: {e}")

@app.get("/health")
def health():
    try:
        _, ver = _load_model()
        return {"status":"ok","model_version": ver}
    except Exception as e:
        return {"status":"error","error": str(e)}

@app.post("/predict")
def predict(feats: Features):
    try:
        pipe, ver = _load_model()
        X = [[getattr(feats, k) for k in FEATURE_NAMES]]
        pred = float(pipe.predict(X)[0])
        return {"prediction": pred, "model_version": ver}
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
