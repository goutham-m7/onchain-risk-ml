"""Synthetic pipeline regression: changing held-out labels must not change decisions."""

import json

from threadpoolctl import threadpool_limits

from risk.graph import contexts
from risk.train import train


def test_training_is_independent_of_future_labels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    steps = {t * 100 + i: t for t in range(1, 50) for i in range(10)}
    graph = contexts(steps, [(n - 1, n) for n in steps if n % 10])
    labels = {n: (-1 if n % 10 == 3 else int(n % 3 == 0)) for n in steps}

    def run():
        with threadpool_limits(limits=1):
            train(steps, labels, graph)
        report = json.loads((tmp_path / "results/evaluation.json").read_text())
        sample = json.loads((tmp_path / "artifacts/requests.json").read_text())
        return report, sample

    before, first_sample = run()
    for n, t in steps.items():
        if t >= 38 and labels[n] != -1:
            labels[n] = 1 - labels[n]
    after, second_sample = run()
    assert first_sample == second_sample
    assert after["splits"]["train"]["unknown"] == 27
    for name in ["logistic", "boosting"]:
        assert (
            before["models"][name]["policy_threshold"]
            == after["models"][name]["policy_threshold"]
        )
        assert (
            before["models"][name]["test"]["tp"] != after["models"][name]["test"]["tp"]
        )
