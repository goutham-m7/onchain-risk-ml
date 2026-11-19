"""Synthetic contract fixtures, not evidence of model quality."""

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from risk.api import app_for, narrate
from risk.drift import drift, reference
from risk.graph import Context, vector
from risk.train import calibrated


@pytest.fixture
def artifact():
    x = np.array(
        [vector(Context(parents=[(1, 1)] * i, prior_mean=0.75)) for i in range(8)]
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = LogisticRegression().fit(x, y)
    sigmoid = LogisticRegression().fit(model.decision_function(x).reshape(-1, 1), y)
    return {
        "model": model,
        "sigmoid": sigmoid,
        "profiles": reference(x),
        "threshold": 0.4,
    }


def test_score_identical_to_batch_with_disabled_explanation(artifact):
    row = Context(parents=[(1, 1)] * 2, prior_mean=0.75)
    p = calibrated(artifact["model"], artifact["sigmoid"], np.array([vector(row)]))[0]
    with TestClient(app_for(artifact)) as api:
        response = api.post("/score", json=row.model_dump())
        assert response.status_code == 200
        result = response.json()
        assert result["probability"] == pytest.approx(p)
        assert result["review"] == (p >= 0.4)
        assert result["explanation"] == {"status": "disabled"}
        assert api.get("/drift").json() == {"n": 1, "status": "insufficient", "psi": {}}


@pytest.mark.parametrize(
    "bad",
    [
        {"parents": [[-1, 1]], "prior_mean": 1},
        {"parents": [], "prior_mean": "NaN"},
        {"parents": [], "prior_mean": 0, "label": 1},
    ],
)
def test_bad_context_is_rejected_and_not_counted(artifact, bad):
    with TestClient(app_for(artifact)) as api:
        assert api.post("/score", json=bad).status_code == 422
        assert api.get("/drift").json()["n"] == 0


def test_constant_reference_detects_shift_and_window_is_bounded(artifact):
    x = np.zeros((128, 8))
    ref = reference(x)
    assert drift(ref, x)["status"] == "ok"
    assert drift(ref, x + 1)["status"] == "alert"
    with TestClient(app_for(artifact, window=64)) as api:
        for _ in range(66):
            api.post(
                "/score", json={"parents": [], "prior_mean": 100}
            ).raise_for_status()
        report = api.get("/drift").json()
        assert report["n"] == 64 and report["status"] == "alert"
        assert report["psi"]["recent_mean_inputs"] > 0.25


def test_local_llm_failure_degrades_only_commentary():
    def fail(req):
        raise httpx.ConnectError("no model running")

    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        assert narrate(client, "demo", [1] * 8, True) == {"status": "unavailable"}


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            "C",
            {
                "status": "selected",
                "text": "Observed parents have mean input count 2 and maximum 2.",
            },
        ),
        ("This is a scam.", {"status": "unavailable"}),
    ],
)
def test_llm_selects_evidence_but_cannot_supply_claims(reply, expected):
    def serve(req):
        assert req.url.host == "127.0.0.1"
        assert "mean_parent_inputs" in req.content.decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        assert narrate(client, "demo", [2] * 8, True) == expected
