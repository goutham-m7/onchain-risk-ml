"""Population stability index with fixed reference bins and empty-bin smoothing."""

import numpy as np

from .graph import NAMES


def reference(x):
    profiles = []
    for c in x.T:
        # Keep underflow and overflow visible, even for a constant reference.
        edges = np.unique(
            np.r_[np.quantile(c, np.arange(10) / 10), np.nextafter(max(c), np.inf)]
        )
        counts = np.bincount(
            np.searchsorted(edges, c, side="right"), minlength=len(edges) + 1
        )
        profiles.append((edges, counts))
    return profiles


def drift(profiles, rows):
    if len(rows) < 64:
        return {"n": len(rows), "status": "insufficient", "psi": {}}
    scores = {}
    for name, c, (edges, expected) in zip(
        NAMES, np.asarray(rows).T, profiles, strict=True
    ):
        observed = np.bincount(
            np.searchsorted(edges, c, side="right"), minlength=len(edges) + 1
        )
        p = (expected + 0.5) / (expected.sum() + 0.5 * len(expected))
        q = (observed + 0.5) / (observed.sum() + 0.5 * len(observed))
        scores[name] = float(np.sum((q - p) * np.log(q / p)))
    return {
        "n": len(rows),
        "status": "alert" if max(scores.values()) >= 0.25 else "ok",
        "psi": scores,
    }
