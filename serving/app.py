import os
import pickle
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Iris MLOps Model Service", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")


def _load_or_train_fallback():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    iris = load_iris()
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(iris.data, iris.target)
    return clf


model = _load_or_train_fallback()


class PredictRequest(BaseModel):
    features: List[List[float]]


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        preds = model.predict(req.features).tolist()
        return {"predictions": preds, "model_version": MODEL_VERSION}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))