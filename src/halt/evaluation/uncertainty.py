"""Bootstrap independent clusters while preserving within-cluster dependence."""
from __future__ import annotations

import math
import random
from collections.abc import Sequence

from halt.errors import ConfigurationError


def cluster_bootstrap(clusters: Sequence[Sequence[float]], *, samples: int = 2000,
                      seed: int = 0) -> tuple[float, float]:
    """Resample clusters, estimating the observation-weighted mean difference.

    Each draw retains all observations in selected clusters. Questions may have
    different numbers of seeds. Clusters themselves must be independent; this
    is not appropriate for questions within an adaptive session.
    """
    if type(samples) is not int or samples < 1 or type(seed) is not int:
        raise ConfigurationError("bootstrap requires positive integer samples and integer seed")
    if len(clusters) < 2 or any(not cluster for cluster in clusters):
        raise ConfigurationError("bootstrap requires at least two nonempty independent clusters")
    if any(isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x)
           for cluster in clusters for x in cluster):
        raise ConfigurationError("cluster values must be finite numbers")
    totals = [(math.fsum(cluster), len(cluster)) for cluster in clusters]
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        draw = rng.choices(totals, k=len(totals))
        values.append(math.fsum(total for total, _ in draw) / sum(n for _, n in draw))
    values.sort()
    return values[math.floor(0.025 * (samples - 1))], values[math.ceil(0.975 * (samples - 1))]
