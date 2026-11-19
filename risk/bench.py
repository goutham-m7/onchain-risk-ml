"""Serial, warm, local HTTP requests with offline/online probability parity checks."""

import argparse
import json
import math
import platform
from pathlib import Path
from time import perf_counter_ns

import httpx
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    rows = json.loads(Path("artifacts/requests.json").read_text())
    latencies = []
    with httpx.Client(
        base_url=f"http://127.0.0.1:{args.port}", trust_env=False, timeout=10
    ) as client:
        for i in range(20 + len(rows) * 4):
            case = rows[(i - 20) % len(rows)]
            start = perf_counter_ns()
            result = client.post("/score", json=case["context"])
            elapsed = (perf_counter_ns() - start) / 1e6
            result.raise_for_status()
            if not math.isclose(
                result.json()["probability"],
                case["expected_probability"],
                abs_tol=1e-10,
            ):
                raise ValueError("batch/online probability mismatch")
            if i >= 20:
                latencies.append(elapsed)
        recent = client.get("/drift")
        recent.raise_for_status()
    report = {
        "transport": "serial HTTP/1.1 loopback, keep-alive, one Uvicorn worker; explanation off",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "warmups": 20,
        "distinct_test_contexts": len(rows),
        "requests": len(latencies),
        "parity_failures": 0,
        "latency_ms": {
            f"p{p}": float(np.percentile(latencies, p)) for p in (50, 95, 99)
        },
        "rolling_drift_after_replaying_labelled_requests": recent.json(),
    }
    Path("results/latency.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
