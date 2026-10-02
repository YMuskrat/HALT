import pytest
from test_comparison_outputs import record

from halt.errors import ConfigurationError
from halt.evaluation import compare
from halt.evaluation.uncertainty import cluster_bootstrap


def test_question_intervals_are_opt_in_and_never_apply_to_adaptive_sessions():
    rows = [record(method, question, seed=seed, correct=(method == 'full_reasoning'))
            for method in ('full_reasoning', 'custom')
            for question in ('q1', 'q2') for seed in (1, 2)]
    assert compare(rows)['methods'][1]['paired_accuracy_ci95'] is None
    result = compare(rows, uncertainty='question', bootstrap_samples=50)['methods'][1]
    assert result['paired_accuracy_ci95'] == (-1, -1)
    assert result['independent_question_count'] == 2
    for row in rows:
        row['session_before'] = {'session_id': 'dependent'}
    assert compare(rows, uncertainty='question')['methods'][1]['paired_accuracy_ci95'] is None


def test_question_interval_requires_independent_clusters_and_valid_option():
    rows = [record(), record('custom')]
    assert compare(rows, uncertainty='question')['methods'][1]['paired_accuracy_ci95'] is None
    with pytest.raises(ConfigurationError, match='uncertainty'):
        compare(rows, uncertainty='unknown')


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
