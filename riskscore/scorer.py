"""Scoring shared by the API (one applicant) and the queue worker (thousands at a time)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from riskscore.features import build_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"

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


@lru_cache(maxsize=1)
def bundle() -> dict:
    return joblib.load(MODEL_PATH)


def applicants_to_frame(applicants: list[dict]) -> pd.DataFrame:
    """Applicant dicts (limit_bal, age, education, pay_status[6], bill_amounts[6], paid_amounts[6]) to raw columns."""
    rows = []
    for a in applicants:
        row = {"limit_bal": a["limit_bal"], "age": a["age"], "education": a["education"]}
        for i in range(6):
            row[f"pay_{i + 1}"] = a["pay_status"][i]
            row[f"bill_{i + 1}"] = a["bill_amounts"][i]
            row[f"paid_{i + 1}"] = a["paid_amounts"][i]
        rows.append(row)
    return pd.DataFrame(rows)


def score_many(applicants: list[dict]) -> list[dict]:
    """Score a batch in one vectorised model call; returns probability, band, decision and top reasons."""
    if not applicants:
        return []
    b = bundle()
    x = build_features(applicants_to_frame(applicants))
    # LightGBM uses every core by default; with several workers on one machine, give each a share instead.
    threads = int(os.getenv("SCORER_THREADS", "0")) or None
    kw = {"num_threads": threads} if threads else {}
    probs = b["model"].predict_proba(x, **kw)[:, 1]
    contrib = b["model"].booster_.predict(x, pred_contrib=True, **kw)[:, :-1]
    names = b["features"]
    out = []
    for p, c in zip(probs, contrib):
        top = np.argsort(-c)[:3]
        out.append({
            "probability_of_default": round(float(p), 4),
            "risk_band": next(name for name, upper in b["bands"] if p < upper),
            "decision": "approve" if p < b["cutoff"] else "decline",
            "cutoff": b["cutoff"],
            "top_reasons": [{"feature": names[i], "description": REASONS[names[i]], "effect": round(float(c[i]), 3)}
                            for i in top if c[i] > 0],
        })
    return out
