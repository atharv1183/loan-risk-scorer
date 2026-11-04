from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

STEADY = {"limit_bal": 200000, "age": 34, "education": 2, "pay_status": [-1] * 6,
          "bill_amounts": [12000, 9800, 15000, 8000, 11000, 9500],
          "paid_amounts": [9800, 15000, 8000, 11000, 9500, 7000]}
BEHIND = {"limit_bal": 50000, "age": 26, "education": 2, "pay_status": [2, 2, 2, 0, 0, 0],
          "bill_amounts": [49500, 48800, 47100, 45000, 43900, 42000],
          "paid_amounts": [0, 0, 1500, 2000, 1800, 1700]}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_steady_payer_is_approved():
    s = client.post("/score", json=STEADY).json()
    assert 0 <= s["probability_of_default"] <= 1
    assert s["decision"] == "approve"


def test_late_payer_is_declined_with_reasons():
    s = client.post("/score", json=BEHIND).json()
    assert s["decision"] == "decline"
    assert s["risk_band"] in {"D", "E"}
    assert any("late" in r["description"].lower() for r in s["top_reasons"])


def test_being_later_never_lowers_risk():
    probs = []
    for months in range(0, 5):
        body = {**STEADY, "pay_status": [months, 0, 0, 0, 0, 0]}
        probs.append(client.post("/score", json=body).json()["probability_of_default"])
    assert probs == sorted(probs)


def test_invalid_input_is_rejected():
    assert client.post("/score", json={**STEADY, "pay_status": [0, 0, 0]}).status_code == 422
    assert client.post("/score", json={**STEADY, "limit_bal": -5}).status_code == 422
    assert client.post("/score", json={**STEADY, "education": 9}).status_code == 422


def test_batch_scoring():
    r = client.post("/score/batch", json=[STEADY, BEHIND])
    assert r.status_code == 200
    assert [s["decision"] for s in r.json()] == ["approve", "decline"]
