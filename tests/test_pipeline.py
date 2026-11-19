import numpy as np
import pytest

from risk.graph import contexts, vector
from risk.train import evaluate, phases, policy_threshold


def test_future_children_cannot_change_past_features():
    steps = {90: 1, 2: 1, 1: 1, 4: 2}
    edges = [(90, 2), (90, 1), (2, 1), (1, 4)]
    rows = contexts(steps, edges)
    assert rows[1].parents == [(0, 0), (1, 1)]
    assert rows[4].parents == [(2, 2)]
    assert rows[4].prior_mean == 1.0
    assert rows[90].prior_mean == 0
    assert contexts({**steps, 8: 3}, edges + [(1, 8)])[4] == rows[4]
    assert vector(rows[1])[:5] == [2, 2, 0.5, 1, 0.5]


def test_trailing_three_completed_steps_only():
    rows = contexts({1: 1, 2: 1, 3: 3, 4: 4, 5: 5}, [(1, 2), (1, 3)])
    assert rows[5].prior_mean == 0.5  # step 1 expired; steps 3 & 4 have two nodes


@pytest.mark.parametrize("edges", [[(2, 1)], [(1, 3)], [(1, 1)], [(1, 2), (1, 2)]])
def test_rejects_impossible_graph(edges):
    with pytest.raises(ValueError):
        contexts({1: 1, 2: 2}, edges)


def test_same_step_cycle_rejected():
    with pytest.raises(ValueError, match="cycle"):
        contexts({1: 1, 2: 1}, [(1, 2), (2, 1)])


def test_threshold_selects_highest_feasible_value_including_ties():
    y = np.array([1, 1, 0, 1, 0])
    p = np.array([0.9, 0.6, 0.6, 0.6, 0.1])
    assert policy_threshold(y, p, 0.8) == 0.6
    metrics = evaluate(y, p, 0.8)
    assert metrics["tp"] == 1 and metrics["fn"] == 2
    assert metrics["retrospective_precision_at_80_recall"] == 0.75


def test_splits_are_disjoint_and_chronological():
    masks = phases(np.array([1, 27, 28, 32, 33, 37, 38, 49]))
    assert [np.flatnonzero(mask).tolist() for mask in masks.values()] == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
    ]
