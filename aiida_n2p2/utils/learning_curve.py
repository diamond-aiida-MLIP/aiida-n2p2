"""Parse and merge n2p2 ``learning-curve.out`` files."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np

# Distinct colours per training run (matplotlib tab10-like).
RUN_COLORS = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
]


@dataclass
class LearningCurveRun:
    """One training run segment parsed from ``learning-curve.out``."""

    run_index: int
    label: str
    color: str
    epoch_offset: int
    epochs_local: np.ndarray
    rmse_train: np.ndarray
    rmse_test: np.ndarray

    @property
    def epochs_global(self) -> np.ndarray:
        return self.epochs_local + self.epoch_offset

    @property
    def n_epochs(self) -> int:
        return int(self.epochs_local.size)


def parse_learning_curve(
    source: str | Path | TextIO | BinaryIO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return epoch, train RMSE/atom, test RMSE/atom arrays."""
    if isinstance(source, Path):
        handle = source.open('r', encoding='utf-8')
        close = True
    elif isinstance(source, str):
        handle = io.StringIO(source)
        close = False
    elif hasattr(source, 'read'):
        handle = source
        close = False
    else:
        raise TypeError(f'Unsupported source type: {type(source)}')

    try:
        data = np.genfromtxt(
            handle,
            comments='#',
            usecols=(0, 1, 2),
            dtype=[('epoch', int), ('rmse_train', float), ('rmse_test', float)],
        )
    finally:
        if close:
            handle.close()

    if data.size == 0:
        empty = np.array([], dtype=int)
        return empty, np.array([]), np.array([])

    if data.ndim == 0:
        data = np.array([data])

    return data['epoch'], data['rmse_train'], data['rmse_test']


def best_epoch_from_curve(
    source: str | Path | TextIO | BinaryIO,
) -> tuple[int, float]:
    """Return best epoch (lowest test RMSE) and the RMSE value."""
    epochs, _, rmse_test = parse_learning_curve(source)
    if epochs.size == 0:
        raise ValueError('Learning curve is empty.')
    index = int(np.argmin(rmse_test))
    return int(epochs[index]), float(rmse_test[index])


def last_epoch_from_curve(
    source: str | Path | TextIO | BinaryIO,
) -> int:
    epochs, _, _ = parse_learning_curve(source)
    if epochs.size == 0:
        raise ValueError('Learning curve is empty.')
    return int(epochs[-1])


def build_run_from_curve(
    source: str | Path | TextIO | BinaryIO,
    *,
    run_index: int,
    epoch_offset: int = 0,
    label: str | None = None,
    color: str | None = None,
) -> LearningCurveRun:
    epochs, rmse_train, rmse_test = parse_learning_curve(source)
    return LearningCurveRun(
        run_index=run_index,
        label=label or f'run-{run_index + 1}',
        color=color or RUN_COLORS[run_index % len(RUN_COLORS)],
        epoch_offset=epoch_offset,
        epochs_local=epochs,
        rmse_train=rmse_train,
        rmse_test=rmse_test,
    )


def merge_learning_curve_runs(runs: list[LearningCurveRun]) -> dict:
    """Merge run segments and build plot-friendly metadata."""
    if not runs:
        raise ValueError('At least one learning-curve run is required.')

    epochs_global: list[int] = []
    epochs_local: list[int] = []
    rmse_train: list[float] = []
    rmse_test: list[float] = []
    run_index_per_point: list[int] = []
    run_metadata: list[dict] = []

    for run in runs:
        run_metadata.append(
            {
                'run_index': run.run_index,
                'label': run.label,
                'color': run.color,
                'epoch_offset': run.epoch_offset,
                'n_epochs': run.n_epochs,
                'epoch_local_start': int(run.epochs_local[0]) if run.n_epochs else None,
                'epoch_local_end': int(run.epochs_local[-1]) if run.n_epochs else None,
                'epoch_global_start': int(run.epochs_global[0]) if run.n_epochs else None,
                'epoch_global_end': int(run.epochs_global[-1]) if run.n_epochs else None,
            }
        )
        for local, global_epoch, train, test in zip(
            run.epochs_local,
            run.epochs_global,
            run.rmse_train,
            run.rmse_test,
            strict=True,
        ):
            epochs_local.append(int(local))
            epochs_global.append(int(global_epoch))
            rmse_train.append(float(train))
            rmse_test.append(float(test))
            run_index_per_point.append(run.run_index)

    return {
        'n_runs': len(runs),
        'runs': run_metadata,
        'epochs_global': epochs_global,
        'epochs_local': epochs_local,
        'run_index': run_index_per_point,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
    }


def render_merged_learning_curve(merged: dict) -> str:
    """Render a merged learning curve file with run provenance in the header."""
    lines = [
        '################################################################################',
        '# Merged learning curve generated by aiida-n2p2',
        '# Each row: global_epoch run_index local_epoch RMSE_train RMSE_test',
        '################################################################################',
        '# run_index  label  color  epoch_offset  n_epochs',
    ]
    for run in merged['runs']:
        lines.append(
            '# {run_index}  {label}  {color}  {epoch_offset}  {n_epochs}'.format(**run)
        )
    lines.append('# global_epoch run_index local_epoch RMSE_train RMSE_test')
    for epoch_g, run_i, epoch_l, train, test in zip(
        merged['epochs_global'],
        merged['run_index'],
        merged['epochs_local'],
        merged['rmse_train'],
        merged['rmse_test'],
        strict=True,
    ):
        lines.append(
            f'{epoch_g:8d} {run_i:9d} {epoch_l:11d} {train:16.8E} {test:16.8E}'
        )
    return '\n'.join(lines) + '\n'


def plot_data_to_runs(plot_data: dict) -> list[LearningCurveRun]:
    """Reconstruct run segments from stored plot metadata."""
    runs: list[LearningCurveRun] = []
    for meta in plot_data['runs']:
        mask = [idx == meta['run_index'] for idx in plot_data['run_index']]
        runs.append(
            LearningCurveRun(
                run_index=meta['run_index'],
                label=meta['label'],
                color=meta['color'],
                epoch_offset=meta['epoch_offset'],
                epochs_local=np.array(
                    [epoch for epoch, keep in zip(plot_data['epochs_local'], mask, strict=True) if keep]
                ),
                rmse_train=np.array(
                    [value for value, keep in zip(plot_data['rmse_train'], mask, strict=True) if keep]
                ),
                rmse_test=np.array(
                    [value for value, keep in zip(plot_data['rmse_test'], mask, strict=True) if keep]
                ),
            )
        )
    return runs
