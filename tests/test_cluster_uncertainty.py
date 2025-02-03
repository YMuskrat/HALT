import pytest

from halt.errors import ConfigurationError
from halt.evaluation.uncertainty import cluster_bootstrap


def test_cluster_bootstrap_preserves_dependence_and_is_deterministic():
    clusters = [[-1.0] * 10, [1.0] * 10]
    result = cluster_bootstrap(clusters, samples=1000, seed=7)
    assert result == (-1.0, 1.0)
    assert result == cluster_bootstrap(clusters, samples=1000, seed=7)
    assert cluster_bootstrap([[0.5], [0.5, 0.5]], samples=20) == (0.5, 0.5)


@pytest.mark.parametrize('clusters', [[], [[1]], [[], [1]], [[float('nan')], [1]], [[True], [1]]])
def test_cluster_bootstrap_rejects_invalid_clusters(clusters):
    with pytest.raises(ConfigurationError):
        cluster_bootstrap(clusters)


@pytest.mark.parametrize('samples', [0, -1, True, 1.5])
def test_cluster_bootstrap_rejects_invalid_samples(samples):
    with pytest.raises(ConfigurationError):
        cluster_bootstrap([[0], [1]], samples=samples)
