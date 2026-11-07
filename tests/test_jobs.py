"""Job queue tests with an in-memory MongoDB (mongomock) and a fake publisher; no RabbitMQ needed."""
import mongomock
import pytest
from fastapi.testclient import TestClient

from api.app import app, get_db, get_publisher
from riskscore import jobs
from riskscore.scorer import score_many
from tests.test_api import BEHIND, STEADY


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, messages):
        self.messages.extend(messages)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(jobs, "CHUNK_SIZE", 2)
    db = mongomock.MongoClient().db
    jobs.ensure_indexes(db)
    pub = FakePublisher()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_publisher] = lambda: pub
    yield TestClient(app), db, pub
    app.dependency_overrides.clear()


def run_worker(db, pub):
    """Process queued messages the way the worker does."""
    for m in pub.messages:
        jobs.process_chunk(db, m["job_id"], m["chunk"], score_many)


def test_job_is_split_into_chunks_and_queued(env):
    client, db, pub = env
    r = client.post("/jobs", json=[STEADY, BEHIND, STEADY, BEHIND, STEADY])
    assert r.status_code == 202
    job = r.json()
    assert job["total_rows"] == 5 and job["total_chunks"] == 3
    assert [m["chunk"] for m in pub.messages] == [0, 1, 2]
    assert client.get(f"/jobs/{job['job_id']}").json()["status"] == "queued"


def test_worker_completes_job_with_summary_and_ordered_results(env):
    client, db, pub = env
    job_id = client.post("/jobs", json=[STEADY, BEHIND, STEADY]).json()["job_id"]
    run_worker(db, pub)
    s = client.get(f"/jobs/{job_id}").json()
    assert s["status"] == "completed"
    assert s["scored_rows"] == 3 and s["chunks_done"] == 2
    assert s["approval_rate"] == pytest.approx(2 / 3, abs=1e-4)
    rows = client.get(f"/jobs/{job_id}/results", params={"offset": 1, "limit": 2}).json()["results"]
    assert [r["row"] for r in rows] == [1, 2]
    assert [r["decision"] for r in rows] == ["decline", "approve"]


def test_redelivered_message_is_not_counted_twice(env):
    client, db, pub = env
    job_id = client.post("/jobs", json=[STEADY, BEHIND]).json()["job_id"]
    assert jobs.process_chunk(db, job_id, 0, score_many) == "done"
    assert jobs.process_chunk(db, job_id, 0, score_many) == "skipped"
    assert db.jobs.find_one({"_id": job_id})["chunks_done"] == 1


def test_failed_chunk_finishes_job_with_errors(env):
    client, db, pub = env
    job_id = client.post("/jobs", json=[STEADY, BEHIND, STEADY]).json()["job_id"]
    jobs.process_chunk(db, job_id, 0, score_many)
    jobs.mark_chunk_failed(db, job_id, 1, "boom")
    s = client.get(f"/jobs/{job_id}").json()
    assert s["status"] == "completed_with_errors"
    assert s["chunks_failed"] == 1 and s["scored_rows"] == 2


def test_queue_down_returns_503_and_marks_job_failed(env):
    client, db, _ = env

    class Down:
        def publish(self, messages):
            raise ConnectionError("broker down")

    app.dependency_overrides[get_publisher] = lambda: Down()
    assert client.post("/jobs", json=[STEADY]).status_code == 503
    assert db.jobs.find_one()["status"] == "failed"


def test_unknown_job_and_bad_input(env):
    client, _, _ = env
    assert client.get("/jobs/nope").status_code == 404
    assert client.post("/jobs", json=[]).status_code == 422
    assert client.post("/jobs", json=[{**STEADY, "age": 5}]).status_code == 422
