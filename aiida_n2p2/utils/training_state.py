"""Helpers for n2p2 training progress and restart eligibility."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np

from aiida_n2p2.utils.input_nn import target_epochs_from_input_nn
from aiida_n2p2.utils.learning_curve import parse_learning_curve

if TYPE_CHECKING:
    from aiida.orm import Node


def resolve_last_training_epoch(
    *,
    curve_last_epoch: int,
    weight_epochs: list[int],
) -> tuple[int, bool]:
    """Return the last completed epoch, preferring remote weight evidence."""
    if not weight_epochs:
        return curve_last_epoch, False
    weight_last = weight_epochs[-1]
    if weight_last > curve_last_epoch:
        return weight_last, True
    return curve_last_epoch, False


def resolve_target_epochs(
    *,
    parameters=None,
    input_nn=None,
) -> int:
    """Read the target ``epochs`` value from structured or file inputs."""
    if parameters is not None:
        keywords = parameters.get_attribute('keywords')
        return int(keywords['epochs']['value'])
    if input_nn is not None:
        return target_epochs_from_input_nn(io.StringIO(input_nn.get_content()))
    raise ValueError('Provide parameters or inputNN to resolve target epochs.')


def training_outputs_available(node: Node) -> bool:
    """Return whether parsed training outputs exist on a process node."""
    try:
        outputs = node.outputs
    except AttributeError:
        return False
    return (
        'last_weights' in outputs
        and 'learning_curve' in outputs
        and 'training_summary' in outputs
    )


def epoch_offset_for_segment(
    previous_global_max: int | None,
    local_epochs: np.ndarray,
) -> int:
    """Choose epoch offset when merging a new learning-curve segment."""
    if previous_global_max is None:
        return 0
    if local_epochs.size and int(local_epochs[0]) > previous_global_max:
        return 0
    return previous_global_max + 1


def global_last_epoch(merged: dict | None, summary: dict) -> int:
    """Return the latest global epoch reached in training."""
    if merged and merged.get('epochs_global'):
        return int(merged['epochs_global'][-1])
    return int(summary['last_epoch'])


def training_reached_target(
    *,
    summary: dict,
    merged: dict | None,
    target_epochs: int,
) -> bool:
    """Return whether training reached the configured global ``epochs`` target."""
    return global_last_epoch(merged, summary) >= target_epochs


def is_restartable_incomplete(
    node: Node,
    *,
    target_epochs: int,
    merged: dict | None = None,
) -> bool:
    """Return whether a calc node stopped early but can be restarted."""
    if not training_outputs_available(node):
        return False
    summary = node.outputs.training_summary.get_dict()
    return not training_reached_target(
        summary=summary,
        merged=merged,
        target_epochs=target_epochs,
    )


def local_epochs_from_curve_content(content: str) -> np.ndarray:
    epochs, _, _ = parse_learning_curve(content)
    return epochs
