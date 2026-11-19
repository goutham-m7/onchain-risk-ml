"""A node's features depend on ancestors and completed time buckets only."""

from collections import Counter
from graphlib import CycleError, TopologicalSorter
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

Count = Annotated[int, Field(ge=0, le=10**7, strict=True)]


class Context(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    parents: Annotated[list[tuple[Count, Count]], Field(max_length=50_000)]
    prior_mean: Annotated[FiniteFloat, Field(ge=0, le=10**7)]


NAMES = (
    "inputs",
    "depth",
    "mean_parent_inputs",
    "max_parent_inputs",
    "mean_parent_depth",
    "root_parent_fraction",
    "recent_mean_inputs",
    "inputs_minus_recent_mean",
)


def vector(row: Context) -> list[float]:
    n = len(row.parents)
    if n:
        counts, depths = zip(*row.parents)
        shape = [
            n,
            max(depths) + 1,
            sum(counts) / n,
            max(counts),
            sum(depths) / n,
            counts.count(0) / n,
        ]
    else:
        shape = [0] * 6
    return [*shape, row.prior_mean, n - row.prior_mean]


def contexts(steps: dict[int, int], edges) -> dict[int, Context]:
    parents = {node: [] for node in steps}
    incoming = Counter()
    seen = set()
    for a, b in edges:
        if a not in steps or b not in steps or steps[a] > steps[b] or (a, b) in seen:
            raise ValueError("unknown, duplicate or backward edge")
        parents[b].append(a)
        seen.add((a, b))
        incoming[steps[b]] += 1
    size = Counter(steps.values())
    recent = {}
    for t in size:
        prior = range(t - 3, t)
        n = sum(size[s] for s in prior)
        recent[t] = sum(incoming[s] for s in prior) / n if n else 0.0
    result, depth = {}, {}
    try:
        for node in TopologicalSorter(parents).static_order():
            summary = [(len(parents[p]), depth[p]) for p in parents[node]]
            depth[node] = max((d + 1 for _, d in summary), default=0)
            result[node] = Context(parents=summary, prior_mean=recent[steps[node]])
    except CycleError as exc:
        raise ValueError("cycle in transaction inputs") from exc
    return result
