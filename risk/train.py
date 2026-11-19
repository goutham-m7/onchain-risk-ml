"""Fit on known labels, calibrate and select policy in separate later windows."""

import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import load
from .drift import drift, reference
from .graph import NAMES, vector

TARGET = 0.80


def phases(t):
    return {
        name: (t >= start) & (t <= end)
        for name, start, end in [
            ("train", 1, 27),
            ("calibration", 28, 32),
            ("policy", 33, 37),
            ("test", 38, 49),
        ]
    }


def policy_threshold(y, p, target):
    positive = p[y == 1]
    if not 0 < target <= 1 or not len(positive):
        raise ValueError("recall target needs positive labels and 0 < target <= 1")
    k = len(positive) - math.ceil(target * len(positive))
    return float(np.partition(positive, k)[k])


def evaluate(y, p, threshold):
    tn, fp, fn, tp = map(
        int, confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    )
    retrospective = None
    if tp + fn:
        selected = p >= policy_threshold(y, p, TARGET)
        retrospective = float(y[selected].mean())
    return {
        "n": len(y),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "alert_rate": (tp + fp) / len(y),
        "average_precision": float(average_precision_score(y, p)) if tp + fn else None,
        "brier": float(brier_score_loss(y, p, pos_label=1)),
        "retrospective_precision_at_80_recall": retrospective,
    }


def calibrated(model, sigmoid, x):
    return sigmoid.predict_proba(model.decision_function(x).reshape(-1, 1))[:, 1]


def train(steps, labels, graph):
    ids = list(steps)
    x = np.asarray([vector(graph[n]) for n in ids], dtype=np.float64)
    y = np.array([labels[n] for n in ids])
    t = np.array([steps[n] for n in ids])
    raw = phases(t)
    folds = {k: v & (y >= 0) for k, v in raw.items()}
    tr, cal, pol, test = folds.values()
    profiles = reference(
        x[raw["train"]]
    )  # monitoring represents traffic, not label availability
    report = {
        "dataset": "Elliptic Bitcoin (real public labels)",
        "features": NAMES,
        "recall_target": TARGET,
        "splits": {
            k: {
                "nodes": int(m.sum()),
                "licit": int((m & (y == 0)).sum()),
                "illicit": int((m & (y == 1)).sum()),
                "unknown": int((m & (y < 0)).sum()),
            }
            for k, m in raw.items()
        },
        "models": {},
        "all_test_traffic_drift": drift(profiles, x[raw["test"]]),
    }
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "boosting": HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            learning_rate=0.08,
            min_samples_leaf=40,
            l2_regularization=1,
            early_stopping=False,
            random_state=42,
        ),
    }
    for name, model in models.items():
        model.fit(x[tr], y[tr])
        sigmoid = LogisticRegression().fit(
            model.decision_function(x[cal]).reshape(-1, 1), y[cal]
        )
        probability = calibrated(model, sigmoid, x[pol])
        threshold = policy_threshold(y[pol], probability, TARGET)
        test_p = calibrated(model, sigmoid, x[test])
        test_y = y[test]
        raw_test_p = model.predict_proba(x[test])[:, 1]
        report["models"][name] = {
            "policy_threshold": threshold,
            "policy": evaluate(y[pol], probability, threshold),
            "test": evaluate(test_y, test_p, threshold),
            "sigmoid_slope": float(sigmoid.coef_[0, 0]),
            "test_uncalibrated_ap": float(average_precision_score(test_y, raw_test_p)),
            "test_uncalibrated_brier": float(brier_score_loss(test_y, raw_test_p)),
            "test_by_step": {
                int(s): evaluate(test_y[t[test] == s], test_p[t[test] == s], threshold)
                for s in np.unique(t[test])
            },
        }
        if name == "boosting":
            joblib.dump(
                {
                    "model": model,
                    "sigmoid": sigmoid,
                    "threshold": threshold,
                    "profiles": profiles,
                },
                out / "model.joblib",
            )
            rng = np.random.default_rng(42)
            sample = rng.choice(
                np.flatnonzero(test), size=min(256, int(test.sum())), replace=False
            )
            sample_p = calibrated(model, sigmoid, x[sample])
            examples = [
                {
                    "context": graph[ids[i]].model_dump(),
                    "expected_probability": float(p),
                }
                for i, p in zip(sample, sample_p, strict=True)
            ]
            (out / "requests.json").write_text(json.dumps(examples))
    Path("results").mkdir(exist_ok=True)
    Path("results/evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({k: v["test"] for k, v in report["models"].items()}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        train(*load())
