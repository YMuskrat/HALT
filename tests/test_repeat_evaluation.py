import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from halt.backends import ScriptedBackend
from halt.config import load_config
from halt.errors import ConfigurationError
from halt.evaluation.repeats import evaluate_repeats


def configuration():
    config = load_config(Path('configs/benchmark_mcq.yaml'))
    return replace(config, evaluation={**config.evaluation, 'bootstrap_samples': 10,
        'methods': [{'name': 'full_reasoning'}, {'name': 'refrain', 'parameters': {'adaptive': True}}]})


def test_repeats_initialize_separate_sessions_and_resume_without_duplicate_records(tmp_path):
    config = configuration()
    before = copy.deepcopy(config.to_dict())
    factories = []

    def fresh_backend():
        backend = ScriptedBackend()
        factories.append(backend)
        return backend

    summaries = evaluate_repeats(config, tmp_path, seeds=[2, 3], backend_factory=fresh_backend)
    assert len(summaries) == 2 and factories[0] is not factories[1]
    checkpoints = sorted(tmp_path.glob('repeat-*/checkpoint.jsonl'))
    original = [path.read_bytes() for path in checkpoints]
    session_ids = []
    for path in checkpoints:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        first = next(record for record in records if record['session_before'] is not None)
        assert first['session_before']['completed'] == {}
        session_ids.append(first['session_before']['session_id'])
    assert len(set(session_ids)) == 2
    evaluate_repeats(config, tmp_path, seeds=[2, 3], backend_factory=fresh_backend)
    assert [path.read_bytes() for path in checkpoints] == original
    assert config.to_dict() == before
    with pytest.raises(ConfigurationError, match='incompatible'):
        evaluate_repeats(config, tmp_path, seeds=[3, 4], backend_factory=fresh_backend)


@pytest.mark.parametrize('seeds', [[], [1, 1], [-1], [True]])
def test_invalid_repeat_seeds_do_not_create_output(tmp_path, seeds):
    output = tmp_path / 'output'
    with pytest.raises(ConfigurationError, match='seeds'):
        evaluate_repeats(configuration(), output, seeds=seeds)
    assert not output.exists()
