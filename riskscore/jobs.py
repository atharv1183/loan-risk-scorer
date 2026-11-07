"""Asynchronous batch scoring: jobs are split into chunks, queued on RabbitMQ, scored by workers, stored in MongoDB.

Flow:
  POST /jobs  -> job document + one chunk document per CHUNK_SIZE applicants in MongoDB,
                 then one small message per chunk on the durable RabbitMQ queue
  worker      -> takes a chunk message, scores the chunk, stores the results, marks the chunk done
  GET /jobs/{id} -> progress and, when finished, a summary
Chunks are independent, so adding workers scores one big job in parallel.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Protocol

from pymongo import ASCENDING, ReturnDocument
from pymongo.database import Database

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
QUEUE = "scoring.chunks"
DEAD_LETTER_QUEUE = "scoring.chunks.dlq"
MAX_ATTEMPTS = 3

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/%2F")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "loanrisk")


def now() -> datetime:
    return datetime.now(timezone.utc)


class Publisher(Protocol):
    def publish(self, messages: list[dict]) -> None: ...


def declare_queues(channel) -> None:
    """Durable work queue whose rejected messages go to a dead-letter queue for inspection."""
    channel.queue_declare(queue=DEAD_LETTER_QUEUE, durable=True)
    channel.queue_declare(queue=QUEUE, durable=True, arguments={
        "x-dead-letter-exchange": "",
        "x-dead-letter-routing-key": DEAD_LETTER_QUEUE,
    })


class RabbitPublisher:
    """Publishes persistent messages with publisher confirms, so a message is on disk before the API replies."""

    def __init__(self, url: str = RABBITMQ_URL):
        self.url = url

    def publish(self, messages: list[dict]) -> None:
        import pika

        conn = pika.BlockingConnection(pika.URLParameters(self.url))
        try:
            ch = conn.channel()
            declare_queues(ch)
            ch.confirm_delivery()
            props = pika.BasicProperties(delivery_mode=pika.DeliveryMode.Persistent, content_type="application/json")
            for m in messages:
                ch.basic_publish(exchange="", routing_key=QUEUE, body=json.dumps(m), properties=props, mandatory=True)
        finally:
            conn.close()


def mongo_db(url: str = MONGO_URL, name: str = MONGO_DB) -> Database:
    from pymongo import MongoClient

    db = MongoClient(url, serverSelectionTimeoutMS=5000)[name]
    ensure_indexes(db)
    return db


def ensure_indexes(db: Database) -> None:
    db.chunks.create_index([("job_id", ASCENDING), ("index", ASCENDING)], unique=True)


def create_job(db: Database, publisher: Publisher, applicants: list[dict]) -> dict:
    job_id = uuid.uuid4().hex
    chunks = [applicants[i:i + CHUNK_SIZE] for i in range(0, len(applicants), CHUNK_SIZE)]
    job = {"_id": job_id, "status": "queued", "total_rows": len(applicants), "total_chunks": len(chunks),
           "chunks_done": 0, "chunks_failed": 0, "created_at": now(), "started_at": None, "finished_at": None}
    db.jobs.insert_one(job)
    db.chunks.insert_many([{"job_id": job_id, "index": i, "status": "queued", "attempts": 0,
                            "applicants": rows, "results": None} for i, rows in enumerate(chunks)])
    try:
        publisher.publish([{"job_id": job_id, "chunk": i} for i in range(len(chunks))])
    except Exception:
        db.jobs.update_one({"_id": job_id}, {"$set": {"status": "failed", "error": "could not queue job"}})
        raise
    return job


def process_chunk(db: Database, job_id: str, index: int, score_fn) -> str:
    """Score one chunk. Safe to run twice for the same chunk: a finished chunk is skipped.

    Returns "done", "skipped" or "missing".
    """
    chunk = db.chunks.find_one_and_update(
        {"job_id": job_id, "index": index, "status": {"$ne": "done"}},
        {"$set": {"status": "running"}, "$inc": {"attempts": 1}},
        return_document=ReturnDocument.AFTER,
    )
    if chunk is None:
        return "skipped" if db.chunks.count_documents({"job_id": job_id, "index": index}) else "missing"

    db.jobs.update_one({"_id": job_id, "started_at": None}, {"$set": {"started_at": now(), "status": "running"}})
    results = score_fn(chunk["applicants"])
    # Only the update that flips the chunk to "done" counts it, so a redelivered message can't double count.
    flipped = db.chunks.update_one({"_id": chunk["_id"], "status": "running"},
                                   {"$set": {"status": "done", "results": results, "applicants": None}})
    if flipped.modified_count:
        job = db.jobs.find_one_and_update({"_id": job_id}, {"$inc": {"chunks_done": 1}},
                                          return_document=ReturnDocument.AFTER)
        finish_if_complete(db, job)
    return "done"


def mark_chunk_failed(db: Database, job_id: str, index: int, error: str) -> None:
    flipped = db.chunks.update_one({"job_id": job_id, "index": index, "status": {"$ne": "done"}},
                                   {"$set": {"status": "failed", "error": error[:500]}})
    if flipped.modified_count:
        job = db.jobs.find_one_and_update({"_id": job_id}, {"$inc": {"chunks_failed": 1}},
                                          return_document=ReturnDocument.AFTER)
        finish_if_complete(db, job)


def finish_if_complete(db: Database, job: dict | None) -> None:
    if job and job["chunks_done"] + job["chunks_failed"] >= job["total_chunks"]:
        status = "completed" if job["chunks_failed"] == 0 else "completed_with_errors"
        db.jobs.update_one({"_id": job["_id"], "finished_at": None},
                           {"$set": {"status": status, "finished_at": now()}})


def job_summary(db: Database, job_id: str) -> dict | None:
    job = db.jobs.find_one({"_id": job_id})
    if job is None:
        return None
    out = {k: job[k] for k in ("status", "total_rows", "total_chunks", "chunks_done", "chunks_failed")}
    out["job_id"] = job_id
    for k in ("created_at", "started_at", "finished_at"):
        out[k] = job[k].isoformat() if job.get(k) else None
    if job.get("started_at") and job.get("finished_at"):
        secs = (job["finished_at"] - job["started_at"]).total_seconds()
        out["processing_seconds"] = round(secs, 2)
        out["rows_per_second"] = round(job["total_rows"] / secs, 1) if secs > 0 else None
    if job["status"].startswith("completed"):
        pipeline = [{"$match": {"job_id": job_id, "status": "done"}}, {"$unwind": "$results"},
                    {"$group": {"_id": "$results.risk_band", "n": {"$sum": 1},
                                "approved": {"$sum": {"$cond": [{"$eq": ["$results.decision", "approve"]}, 1, 0]}}}}]
        bands = {r["_id"]: r for r in db.chunks.aggregate(pipeline)}
        scored = sum(r["n"] for r in bands.values())
        out["scored_rows"] = scored
        out["approval_rate"] = round(sum(r["approved"] for r in bands.values()) / scored, 4) if scored else None
        out["risk_bands"] = {k: bands[k]["n"] for k in sorted(bands)}
    return out


def job_results(db: Database, job_id: str, offset: int, limit: int) -> list[dict]:
    """Results in input order, paged by row."""
    first, last = offset // CHUNK_SIZE, (offset + limit - 1) // CHUNK_SIZE
    rows = []
    for c in db.chunks.find({"job_id": job_id, "index": {"$gte": first, "$lte": last}}).sort("index", ASCENDING):
        base = c["index"] * CHUNK_SIZE
        for i, r in enumerate(c.get("results") or []):
            if offset <= base + i < offset + limit:
                rows.append({"row": base + i, **r})
    return rows
