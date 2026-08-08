"""Tests for training progress helpers and auto-restart logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aiida_n2p2.utils.input_nn import target_epochs_from_input_nn
from aiida_n2p2.utils.training_state import (
    epoch_offset_for_segment,
    global_last_epoch,
    is_restartable_incomplete,
    resolve_last_training_epoch,
    training_reached_target,
)

EXAMPLE_NN = Path(__file__).parent.parent / 'examples' / '1.Al' / 'input.nn'


def test_resolve_last_training_epoch_prefers_remote_weights():
    last_epoch, used_weights = resolve_last_training_epoch(
        curve_last_epoch=76,
        weight_epochs=[0, 10, 79],
    )
    assert last_epoch == 79
    assert used_weights is True


def test_resolve_last_training_epoch_uses_curve_when_weights_match():
    last_epoch, used_weights = resolve_last_training_epoch(
        curve_last_epoch=76,
        weight_epochs=[0, 76],
    )
    assert last_epoch == 76
    assert used_weights is False


def test_target_epochs_from_input_nn():
    assert target_epochs_from_input_nn(EXAMPLE_NN) == 200


def test_epoch_offset_when_local_epochs_restart_from_one():
    local = np.array([1, 2, 3])
    assert epoch_offset_for_segment(50, local) == 51


def test_epoch_offset_when_local_epochs_are_global():
    local = np.array([51, 52, 53])
    assert epoch_offset_for_segment(50, local) == 0


def test_training_reached_target_uses_merged_global_epochs():
    summary = {'last_epoch': 20, 'training_completed': False}
    merged = {'epochs_global': list(range(1, 41))}
    assert training_reached_target(
        summary=summary,
        merged=merged,
        target_epochs=40,
    )
    assert not training_reached_target(
        summary=summary,
        merged=merged,
        target_epochs=41,
    )


def test_training_reached_target_ignores_per_segment_completion_flag():
    summary = {'last_epoch': 80, 'training_completed': True}
    merged = {'epochs_global': list(range(1, 21))}
    assert not training_reached_target(
        summary=summary,
        merged=merged,
        target_epochs=80,
    )


def test_global_last_epoch_prefers_merged_data():
    summary = {'last_epoch': 10}
    merged = {'epochs_global': [1, 2, 3, 4, 5]}
    assert global_last_epoch(merged, summary) == 5


def test_is_restartable_incomplete_with_failed_calc_node(
    incomplete_train_calcjob_node,
):
    assert is_restartable_incomplete(
        incomplete_train_calcjob_node,
        target_epochs=200,
    )
