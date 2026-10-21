"""Finite, cluster-weighted descriptive estimates; no model dependencies."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping


def percentile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("nonempty values and probability in [0,1] required")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def cluster_estimate(rows: Iterable[Mapping[str, Any]], *, value: str, cluster: str,
                     seed: int, draws: int) -> dict[str, Any]:
    """Average within clusters then bootstrap equally weighted clusters.

    Missing/nonfinite observations are reported rather than counted as zero.
    One cluster has a point estimate but cannot establish uncertainty.
    """
    if type(seed) is not int or type(draws) is not int or draws < 100:
        raise ValueError("integer seed and at least 100 bootstrap draws required")
    groups = defaultdict(list)
    excluded = defaultdict(int)
    for row in rows:
        item = row.get(value)
        if not row.get(cluster):
            excluded["missing_cluster"] += 1
        elif not isinstance(item, (int, float, bool)) or not math.isfinite(item):
            excluded["missing_or_nonfinite_value"] += 1
        else:
            groups[str(row[cluster])].append(float(item))
    means = [sum(groups[key]) / len(groups[key]) for key in sorted(groups)]
    interval = None
    if len(means) >= 2:
        rng = random.Random(seed)
        samples = [sum(rng.choice(means) for _ in means) / len(means) for _ in range(draws)]
        interval = [percentile(samples, .025), percentile(samples, .975)]
    return {"mean": sum(means) / len(means) if means else None,
            "ci95": interval, "n_clusters": len(means),
            "n_rows": sum(map(len, groups.values())), "cluster": cluster,
            "excluded": dict(excluded), "bootstrap_seed": seed, "bootstrap_draws": draws,
            "uncertainty": "percentile_cluster_bootstrap" if interval else "insufficient_clusters"}
