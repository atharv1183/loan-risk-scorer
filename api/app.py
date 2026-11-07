"""Scoring API: send an applicant's account history, get default probability, risk band, decision and reasons.

Run:  uvicorn api.app:app --reload   then open http://127.0.0.1:8000
Large batches go through the job queue (POST /jobs); see docker-compose.yml to run RabbitMQ, MongoDB and workers.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from riskscore import jobs
from riskscore.scorer import bundle, score_many

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "reports" / "metrics.json"
MAX_JOB_ROWS = 100_000

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


class JobCreated(BaseModel):
    job_id: str
    status: str
    total_rows: int
    total_chunks: int


@lru_cache(maxsize=1)
def get_db():
    return jobs.mongo_db()


@lru_cache(maxsize=1)
def get_publisher() -> jobs.Publisher:
    return jobs.RabbitPublisher()


app = FastAPI(title="Loan Risk Scorer", version="1.1.0")


@app.get("/health")
def health() -> dict:
    b = bundle()
    return {"status": "ok", "model_test_auc": b["test_auc"], "features": len(b["features"])}


@app.post("/score", response_model=Score)
def score(applicant: Applicant) -> dict:
    return score_many([applicant.model_dump()])[0]


@app.post("/score/batch", response_model=list[Score])
def score_batch(applicants: list[Applicant]) -> list[dict]:
    """Synchronous batch for small lists; use /jobs for large ones."""
    if len(applicants) > 1000:
        raise HTTPException(413, "Up to 1,000 applicants here; send larger batches to POST /jobs")
    return score_many([a.model_dump() for a in applicants])


@app.post("/jobs", response_model=JobCreated, status_code=202)
def create_job(applicants: list[Applicant], db=Depends(get_db), publisher=Depends(get_publisher)) -> dict:
    """Queue a large batch. Returns at once with a job id; poll GET /jobs/{id} for progress."""
    if not applicants:
        raise HTTPException(422, "Send at least one applicant")
    if len(applicants) > MAX_JOB_ROWS:
        raise HTTPException(413, f"Up to {MAX_JOB_ROWS:,} applicants per job")
    try:
        job = jobs.create_job(db, publisher, [a.model_dump() for a in applicants])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, "Job queue unavailable") from exc
    return {"job_id": job["_id"], **{k: job[k] for k in ("status", "total_rows", "total_chunks")}}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, db=Depends(get_db)) -> dict:
    summary = jobs.job_summary(db, job_id)
    if summary is None:
        raise HTTPException(404, "No such job")
    return summary


@app.get("/jobs/{job_id}/results")
def get_job_results(job_id: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000),
                    db=Depends(get_db)) -> dict:
    if db.jobs.find_one({"_id": job_id}, {"_id": 1}) is None:
        raise HTTPException(404, "No such job")
    return {"job_id": job_id, "offset": offset, "results": jobs.job_results(db, job_id, offset, limit)}


@app.get("/metrics")
def metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(Path(__file__).parent / "dashboard.html")
