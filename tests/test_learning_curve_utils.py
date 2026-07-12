"""Tests for learning-curve parsing and merging."""

from __future__ import annotations

import io
from pathlib import Path

from aiida_n2p2.utils.learning_curve import (
    build_run_from_curve,
    merge_learning_curve_runs,
    plot_data_to_runs,
)

FIXTURES_AL = Path(__file__).parent / 'fixtures' / 'al'


def test_merge_two_runs_offsets_epochs():
    lc = FIXTURES_AL / 'learning-curve.out'
    run1 = build_run_from_curve(lc, run_index=0, epoch_offset=0, label='run-1')
    run2 = build_run_from_curve(
        lc,
        run_index=1,
        epoch_offset=int(run1.epochs_local[-1]) + 1,
        label='run-2 (restart)',
    )
    merged = merge_learning_curve_runs([run1, run2])

    assert merged['n_runs'] == 2
    assert merged['runs'][1]['epoch_offset'] == 201
    assert merged['epochs_global'][0] == 0
    assert merged['epochs_global'][201] == 201
    assert merged['run_index'][0] == 0
    assert merged['run_index'][201] == 1
    assert merged['runs'][0]['color'] != merged['runs'][1]['color']


def test_plot_data_roundtrip():
    lc = FIXTURES_AL / 'learning-curve.out'
    run = build_run_from_curve(lc, run_index=0)
    merged = merge_learning_curve_runs([run])
    rebuilt = plot_data_to_runs(merged)
    assert len(rebuilt) == 1
    assert rebuilt[0].n_epochs == run.n_epochs
