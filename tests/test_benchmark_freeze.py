from dataclasses import replace
from pathlib import Path

import pytest

from halt.config import load_config
from halt.errors import ConfigurationError
from halt.evaluation.freeze import benchmark_identity, freeze_benchmark, verify_benchmark


def test_freeze_rejects_overwrite_and_changed_inputs(tmp_path):
    config = load_config('configs/benchmark_mcq.yaml')
    dataset = tmp_path / 'dataset.jsonl'
    dataset.write_bytes(Path(config.evaluation['dataset']).read_bytes())
    config = replace(config, evaluation={**config.evaluation, 'dataset': str(dataset)})
    lock = tmp_path / 'lock.json'
    identity = freeze_benchmark(config, lock)
    assert identity['selected_dataset']['ordered_item_ids']
    verify_benchmark(config, lock)
    with pytest.raises(FileExistsError):
        freeze_benchmark(config, lock)
    changed = replace(config, runtime={**config.runtime, 'seed': 99})
    with pytest.raises(ConfigurationError, match='differ'):
        verify_benchmark(changed, lock)
    dataset.write_bytes(dataset.read_bytes() + b'\n')
    with pytest.raises(ConfigurationError, match='differ'):
        verify_benchmark(config, lock)


def test_floating_model_revision_is_rejected_before_loading_weights():
    config = load_config('configs/benchmark_mcq.yaml')
    config = replace(config, backend={'name': 'transformers'}, model={'revision': 'main'})
    with pytest.raises(ConfigurationError, match='immutable'):
        benchmark_identity(config)
