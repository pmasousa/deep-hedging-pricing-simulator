"""API contract gates: schema, input-box enforcement, determinism — Sprint C.

Runs on the ``test`` budget (tiny nets) — contract behavior, not model
quality; accuracy lives in test_train.py and reports/benchmarks.
"""

import os
import warnings

os.environ.setdefault("DHPS_API_BUDGET", "test")
# fastapi.testclient re-exports starlette's shim, which warns on import —
# dependency noise, not ours
warnings.filterwarnings("ignore", message="Using `httpx` with "
                        "`starlette.testclient`", category=Warning)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from dhps.api.app import create_app  # noqa: E402

ATM = {"spot": 100.0, "strike": 100.0, "t_maturity": 1.0, "sigma": 0.2}


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:  # context manager runs the lifespan
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_meta_discloses_models_and_box(client):
    meta = client.get("/meta").json()
    assert meta["budget"] == "test"
    assert meta["constants"]["r"] == 0.05 and meta["constants"]["q"] == 0.01
    assert meta["input_box"]["spot"] == [50.0, 200.0]
    assert "price_mae" in meta["pricer"]["metrics"]
    assert "cvar95" in meta["hedge_policy"]["metrics"]


def test_price_contract(client):
    r = client.post("/price", json=ATM)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"price"}
    # loose sanity for the tiny budget: a call price is positive and below
    # the spot it pays on
    assert 0.0 < body["price"] < 200.0


def test_price_determinism(client):
    a = client.post("/price", json=ATM).text
    b = client.post("/price", json=ATM).text
    assert a == b


def test_greeks_contract(client):
    body = client.post("/greeks", json=ATM).json()
    assert set(body) == {"price", "delta", "gamma", "vega", "theta",
                         "dual_delta"}
    assert all(isinstance(v, float) for v in body.values())


def test_hedge_contract(client):
    q = {"spot": 100.0, "strike": 100.0, "time_to_maturity": 0.5,
         "position": 0.5}
    body = client.post("/hedge", json=q).json()
    assert set(body) == {"target_position"}
    assert 0.0 <= body["target_position"] <= 1.0


def test_box_violations_rejected(client):
    cases = [
        {**ATM, "spot": 250.0},           # outside spot box
        {**ATM, "sigma": 0.9},            # outside vol box
        {**ATM, "t_maturity": 3.0},       # outside maturity box
        {**ATM, "strike": 200.0},         # moneyness 2.0 outside box
    ]
    for q in cases:
        r = client.post("/price", json=q)
        assert r.status_code == 422, (q, r.status_code)
        assert "outside training box" in str(r.json()["detail"])


def test_structural_violations_rejected(client):
    assert client.post("/price", json={**ATM, "spot": -1.0}).status_code == 422
    assert client.post("/hedge", json={"spot": 100.0, "strike": 100.0,
                                       "time_to_maturity": 1.5,
                                       "position": 0.5}).status_code == 422
    assert client.post("/hedge", json={"spot": 100.0, "strike": 100.0,
                                       "time_to_maturity": 0.5,
                                       "position": 1.5}).status_code == 422
