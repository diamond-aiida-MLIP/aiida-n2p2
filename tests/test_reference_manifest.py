"""Fast checks: fixture files match REFERENCE.json (no AiiDA DB required)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

FIXTURES_AL = Path(__file__).parent / 'fixtures' / 'al'
REFERENCE_PATH = FIXTURES_AL / 'REFERENCE.json'


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture(scope='module')
def reference() -> dict:
    with REFERENCE_PATH.open(encoding='utf-8') as handle:
        return json.load(handle)


def test_reference_json_exists(reference):
    assert reference['element'] == 'Al'
    assert reference['atomic_number'] == 13
    assert reference['source_workchain_pk'] == 949


def test_scaling_fixture_matches_reference(reference):
    scaling_file = FIXTURES_AL / reference['scaling']['file']
    assert scaling_file.is_file(), f'Missing fixture: {scaling_file}'
    assert _md5(scaling_file) == reference['scaling']['md5']
    assert scaling_file.stat().st_size == reference['scaling']['size_bytes']


def test_best_weights_fixture_matches_reference(reference):
    weights_file = FIXTURES_AL / reference['training']['best_weights_file']
    assert weights_file.is_file(), f'Missing fixture: {weights_file}'
    assert _md5(weights_file) == reference['training']['best_weights_md5']


def test_learning_curve_fixture_exists(reference):
    lc_file = FIXTURES_AL / reference['training']['learning_curve_file']
    assert lc_file.is_file(), f'Missing fixture: {lc_file}'


def test_regression_manifest_human_readable(reference, capsys):
    """Prints the golden reference when run with ``pytest -s``."""
    print('\n--- Golden reference (Al, dump-949) ---')
    print(json.dumps(reference, indent=2))
    print('--- end ---\n')
    assert reference['training']['best_epoch'] == 182
