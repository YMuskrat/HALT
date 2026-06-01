import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from halt.backends import ScriptedBackend
from halt.config import load_config
from halt.errors import ConfigurationError
from halt.evaluation.repeats import evaluate_repeats, summarize_repeats


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
    combined = summarize_repeats(summaries, samples=20)
    assert combined['seeds'] == [2, 3]
    assert combined['experiment']['evidence_kind'] == 'simulated'
    assert all(m['uncertainty_unit'] == 'independent_repeat' for m in combined['methods'])
    with pytest.raises(ConfigurationError, match='distinct seed'):
        summarize_repeats([summaries[0], summaries[0]])
    incompatible = copy.deepcopy(summaries)
    incompatible[1]['comparison_id'] = 'different'
    with pytest.raises(ConfigurationError, match='incompatible'):
        summarize_repeats(incompatible)
    incompatible = copy.deepcopy(summaries)
    incompatible[1]['repeat_group_id'] = 'different-method-parameters'
    with pytest.raises(ConfigurationError, match='incompatible'):
        summarize_repeats(incompatible)
    incomplete = copy.deepcopy(summaries)
    incomplete[1]['methods'][0]['unpaired_sample_count'] = 1
    report = summarize_repeats(incomplete)
    method = next(m for m in report['methods'] if m['method_id'] == incomplete[1]['methods'][0]['method_id'])
    assert method['paired_accuracy_ci95'] is None
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


def test_changed_dataset_and_interrupted_manifest_cannot_resume(tmp_path):
    config = configuration()
    dataset = tmp_path / 'dataset.jsonl'
    dataset.write_bytes(Path(config.evaluation['dataset']).read_bytes())
    config = replace(config, evaluation={**config.evaluation, 'dataset': str(dataset)})
    output = tmp_path / 'runs'
    evaluate_repeats(config, output, seeds=[1], backend_factory=ScriptedBackend)
    checkpoint = (output / 'repeat-0000/checkpoint.jsonl').read_bytes()
    dataset.write_bytes(dataset.read_bytes() + b'\n')
    with pytest.raises(ConfigurationError, match='incompatible'):
        evaluate_repeats(config, output, seeds=[1], backend_factory=ScriptedBackend)
    assert (output / 'repeat-0000/checkpoint.jsonl').read_bytes() == checkpoint
    (output / 'repeats.json').write_text('{incomplete')
    with pytest.raises(ConfigurationError, match='invalid repeat manifest'):
        evaluate_repeats(config, output, seeds=[1], backend_factory=ScriptedBackend)


def test_unrelated_output_is_never_overwritten(tmp_path):
    (tmp_path / 'notes.txt').write_text('keep me')
    with pytest.raises(ConfigurationError, match='empty'):
        evaluate_repeats(configuration(), tmp_path, seeds=[1])
    assert (tmp_path / 'notes.txt').read_text() == 'keep me'
