import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download

app = FastAPI(title="LEWIF ML Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_bundle = joblib.load(
    hf_hub_download(repo_id="ParamFpv/lewif-crp-model", filename="crp_model.joblib")
)
_model = _bundle["model"]

FEATURES = [
    "age", "female", "bmi", "ever_smoked", "sleep_hours",
    "trouble_sleeping", "vigorous_work", "vigorous_recreation",
    "sedentary_minutes", "ever_drinks",
]


class CRPInput(BaseModel):
    age: float = Field(..., ge=0, le=120)
    female: int = Field(..., ge=0, le=1)
    bmi: float = Field(..., ge=10, le=80)
    ever_smoked: int = Field(..., ge=0, le=1)
    sleep_hours: float = Field(..., ge=0, le=24)
    trouble_sleeping: int = Field(..., ge=0, le=1)
    vigorous_work: int = Field(..., ge=0, le=1)
    vigorous_recreation: int = Field(..., ge=0, le=1)
    sedentary_minutes: float = Field(..., ge=0, le=1440)
    ever_drinks: int = Field(..., ge=0, le=1)


class CRPOutput(BaseModel):
    crp_mg_per_l: float
    model_mae: float
    model_r2: float


@app.get("/")
def health():
    return {"status": "ok", "service": "lewif-ml-service"}


@app.post("/predict/crp", response_model=CRPOutput)
def predict_crp(data: CRPInput):
    row = pd.DataFrame([data.model_dump()], columns=FEATURES)
    crp = float(np.expm1(_model.predict(row)[0]))
    return CRPOutput(
        crp_mg_per_l=round(max(0.1, crp), 2),
        model_mae=_bundle.get("mae", 0),
        model_r2=_bundle.get("r2", 0),
    )
