"""Read-only review scores. Start with: uv run uvicorn risk.api:create_app --factory"""

import json
import os
from collections import deque
from contextlib import asynccontextmanager
from threading import Lock

import httpx
import joblib
import numpy as np
from fastapi import FastAPI
from threadpoolctl import threadpool_limits

from .drift import drift
from .graph import NAMES, Context, vector
from .train import calibrated


def narrate(client, model_name, values, review):
    # Fixed loopback destination. Never send IDs, labels or raw text to a hosted model.
    facts = dict(zip(NAMES, values, strict=True))
    payload = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": 5,
        "messages": [
            {
                "role": "system",
                "content": "Choose one observation to show a transaction reviewer. A = input count vs recent mean. B = ancestry depth. C = parent input counts. Reply with a single letter: A, B, or C.",
            },
            {
                "role": "user",
                "content": json.dumps({"features": facts, "review_flag": review}),
            },
        ],
    }
    try:
        response = client.post(
            "http://127.0.0.1:18492/v1/chat/completions", json=payload
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        evidence = {
            "A": f"Observed input count {facts['inputs']}; prior three-step mean {facts['recent_mean_inputs']:.3g}.",
            "B": f"Longest observed upstream path: {facts['depth']} edges.",
            "C": (
                f"Observed parents have mean input count {facts['mean_parent_inputs']:.3g} and maximum {facts['max_parent_inputs']}."
                if facts["inputs"]
                else "No parent transactions are observed in the sampled graph."
            ),
        }
        if isinstance(text, str) and text.strip() in evidence:
            return {"status": "selected", "text": evidence[text.strip()]}
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        pass
    return {"status": "unavailable"}


def app_for(artifact, *, model_name=None, window=512):
    rows = deque(maxlen=window)
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        # Explicitly avoid ambient proxy settings, redirects and remote fallback.
        with (
            httpx.Client(timeout=3, trust_env=False) as client,
            threadpool_limits(limits=1),
        ):
            app.state.llm = client
            yield

    app = FastAPI(title="Bitcoin graph review score (research)", lifespan=lifespan)

    @app.post("/score")
    def score(row: Context, explain: bool = False):
        values = vector(row)
        probability = float(
            calibrated(artifact["model"], artifact["sigmoid"], np.array([values]))[0]
        )
        review = probability >= artifact["threshold"]
        with lock:
            rows.append(values)
        note = (
            narrate(app.state.llm, model_name, values, review)
            if explain and model_name
            else {"status": "disabled"}
        )
        return {
            "probability": probability,
            "review": review,
            "threshold": artifact["threshold"],
            "explanation": note,
        }

    @app.get("/drift")
    def monitor():
        with lock:
            sample = list(rows)
        return drift(artifact["profiles"], sample)

    return app


def create_app():
    # joblib can execute code. Only load an artifact produced by the local training command.
    return app_for(
        joblib.load("artifacts/model.joblib"),
        model_name=os.environ.get("RISK_LOCAL_LLM"),
    )
