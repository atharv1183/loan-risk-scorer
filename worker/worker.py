"""Queue worker: consumes chunk messages from RabbitMQ, scores them, writes results to MongoDB.

Run:  python -m worker.worker        (start several to score in parallel)

- prefetch_count=1: each worker holds one chunk at a time, so work spreads evenly.
- A message is acknowledged only after its results are saved; if a worker dies mid-chunk,
  RabbitMQ hands the chunk to another worker.
- A failing chunk is retried up to MAX_ATTEMPTS times, then sent to the dead-letter queue.
"""
from __future__ import annotations

import json
import logging
import signal
import time

import pika

from riskscore.jobs import MAX_ATTEMPTS, QUEUE, RABBITMQ_URL, declare_queues, mark_chunk_failed, mongo_db, process_chunk
from riskscore.scorer import bundle, score_many

log = logging.getLogger("worker")


def handle(db, ch, method, properties, body) -> None:
    msg = json.loads(body)
    job_id, index = msg["job_id"], int(msg["chunk"])
    attempts = int((properties.headers or {}).get("x-attempts", 1))
    started = time.perf_counter()
    try:
        outcome = process_chunk(db, job_id, index, score_many)
        log.info("job %s chunk %d %s in %.0f ms", job_id[:8], index, outcome, 1000 * (time.perf_counter() - started))
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as exc:  # noqa: BLE001 - any failure is retried, then dead-lettered
        log.exception("job %s chunk %d failed (attempt %d)", job_id[:8], index, attempts)
        if attempts < MAX_ATTEMPTS:
            db.chunks.update_one({"job_id": job_id, "index": index, "status": "running"}, {"$set": {"status": "queued"}})
            ch.basic_publish(exchange="", routing_key=QUEUE, body=body, properties=pika.BasicProperties(
                delivery_mode=pika.DeliveryMode.Persistent, headers={"x-attempts": attempts + 1}))
            ch.basic_ack(delivery_tag=method.delivery_tag)
        else:
            mark_chunk_failed(db, job_id, index, repr(exc))
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)  # goes to the dead-letter queue


def connect(url: str, tries: int = 30) -> pika.BlockingConnection:
    for i in range(tries):
        try:
            return pika.BlockingConnection(pika.URLParameters(url))
        except pika.exceptions.AMQPConnectionError:
            log.info("waiting for RabbitMQ (%d/%d)", i + 1, tries)
            time.sleep(2)
    raise SystemExit("RabbitMQ not reachable")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    bundle()  # load the model once, before taking work
    db = mongo_db()
    conn = connect(RABBITMQ_URL)
    ch = conn.channel()
    declare_queues(ch)
    ch.basic_qos(prefetch_count=1)
    ch.basic_consume(queue=QUEUE, on_message_callback=lambda c, m, p, b: handle(db, c, m, p, b))

    def stop(*_):
        log.info("stopping after the current chunk")
        ch.stop_consuming()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log.info("worker ready on queue %s", QUEUE)
    ch.start_consuming()
    conn.close()


if __name__ == "__main__":
    main()
