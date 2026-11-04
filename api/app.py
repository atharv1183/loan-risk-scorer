"""Scoring API: send an applicant's account history, get default probability, risk band, decision and reasons.

Run:  uvicorn api.app:app --reload   then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from riskscore.features import build_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"

REASONS = {
    "max_months_late": "Worst payment delay in the last 6 months",
    "months_late_1": "Late on the most recent payment",
    "months_late_2": "Late on the payment 2 months ago",
    "months_late_3": "Late on the payment 3 months ago",
    "times_late_6m": "Number of late months in the last 6",
    "late_streak": "Consecutive recent late months",
    "util_1": "High balance compared with credit limit",
    "util_avg_6m": "High average balance compared with limit",
    "util_trend": "Balance rising against the limit",
    "pay_ratio_1": "Small last payment compared with the bill",
    "pay_ratio_avg": "Small payments compared with bills",
    "zero_payment_months": "Months with a bill but no payment",
    "limit_bal": "Credit limit",
    "bill_avg_6m": "Average monthly bill",
    "paid_avg_6m": "Average monthly payment",
    "age": "Age",
    "education": "Education level",
}

Status = Annotated[int, Field(ge=-2, le=9, description="-2 no use, -1 paid in full, 0 revolving, 1-9 months late")]
Amount = Annotated[float, Field(ge=-1e7, le=1e8)]


class Applicant(BaseModel):
    limit_bal: float = Field(gt=0, le=1e8, description="Credit limit")
    age: int = Field(ge=18, le=100)
    education: Literal[1, 2, 3, 4] = Field(description="1 graduate school, 2 university, 3 high school, 4 other")
    pay_status: list[Status] = Field(min_length=6, max_length=6, description="Most recent month first")
    bill_amounts: list[Amount] = Field(min_length=6, max_length=6, description="Most recent month first")
    paid_amounts: list[Annotated[float, Field(ge=0, le=1e8)]] = Field(min_length=6, max_length=6,
                                                                       description="Most recent month first")


class Reason(BaseModel):
    feature: str
    description: str
    effect: float  # contribution to the log-odds of default; positive raises risk


class Score(BaseModel):
    probability_of_default: float
    risk_band: str
    decision: Literal["approve", "decline"]
    cutoff: float
    top_reasons: list[Reason]


@lru_cache(maxsize=1)
def bundle() -> dict:
    return joblib.load(MODEL_PATH)


def to_frame(a: Applicant) -> pd.DataFrame:
    row = {"limit_bal": a.limit_bal, "age": a.age, "education": a.education}
    for i in range(6):
        row[f"pay_{i + 1}"] = a.pay_status[i]
        row[f"bill_{i + 1}"] = a.bill_amounts[i]
        row[f"paid_{i + 1}"] = a.paid_amounts[i]
    return pd.DataFrame([row])


def score_applicant(a: Applicant) -> Score:
    b = bundle()
    x = build_features(to_frame(a))
    p = float(b["model"].predict_proba(x)[0, 1])
    contrib = b["model"].booster_.predict(x, pred_contrib=True)[0, :-1]
    order = np.argsort(-contrib)[:3]
    reasons = [Reason(feature=b["features"][i], description=REASONS[b["features"][i]], effect=round(float(contrib[i]), 3))
               for i in order if contrib[i] > 0]
    band = next(name for name, upper in b["bands"] if p < upper)
    return Score(probability_of_default=round(p, 4), risk_band=band,
                 decision="approve" if p < b["cutoff"] else "decline", cutoff=b["cutoff"], top_reasons=reasons)


app = FastAPI(title="Loan Risk Scorer", version="1.0.0")


@app.get("/health")
def health() -> dict:
    b = bundle()
    return {"status": "ok", "model_test_auc": b["test_auc"], "features": len(b["features"])}


@app.post("/score", response_model=Score)
def score(applicant: Applicant) -> Score:
    return score_applicant(applicant)


@app.post("/score/batch", response_model=list[Score])
def score_batch(applicants: list[Applicant]) -> list[Score]:
    return [score_applicant(a) for a in applicants[:1000]]


@app.get("/metrics")
def metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(Path(__file__).parent / "dashboard.html")
