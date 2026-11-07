"""Submit the whole dataset (30,000 applicants) as one job and time it.

Usage:  python scripts/load_test.py [--url http://localhost:8000] [--rows 30000]
"""
from __future__ import annotations

import argparse
import time

import httpx
import pandas as pd

from riskscore.data import CLEAN_CSV


def applicants(n: int) -> list[dict]:
    df = pd.read_csv(CLEAN_CSV).head(n)
    return [{"limit_bal": float(r.limit_bal), "age": int(r.age), "education": int(r.education),
             "pay_status": [int(getattr(r, f"pay_{i}")) for i in range(1, 7)],
             "bill_amounts": [float(getattr(r, f"bill_{i}")) for i in range(1, 7)],
             "paid_amounts": [float(getattr(r, f"paid_{i}")) for i in range(1, 7)]} for r in df.itertuples()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rows", type=int, default=30000)
    args = ap.parse_args()

    body = applicants(args.rows)
    with httpx.Client(base_url=args.url, timeout=120) as c:
        t0 = time.perf_counter()
        job = c.post("/jobs", json=body).raise_for_status().json()
        queued = time.perf_counter() - t0
        print(f"queued {job['total_rows']:,} rows in {job['total_chunks']} chunks ({queued:.2f}s to accept)")
        while True:
            s = c.get(f"/jobs/{job['job_id']}").json()
            if s["status"].startswith("completed") or s["status"] == "failed":
                break
            time.sleep(0.25)
        total = time.perf_counter() - t0
    print(f"status {s['status']}: {s.get('scored_rows', 0):,} scored, {s['chunks_failed']} chunks failed")
    print(f"worker time {s.get('processing_seconds')}s ({s.get('rows_per_second')} rows/s); end to end {total:.2f}s")
    print(f"approval rate {s.get('approval_rate')}, bands {s.get('risk_bands')}")


if __name__ == "__main__":
    main()
