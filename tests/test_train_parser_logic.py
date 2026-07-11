"""Parser logic tests without submitting calculations."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiida_n2p2.parsers.train import nnpTrainParser

FIXTURES_AL = Path(__file__).parent / 'fixtures' / 'al'


def test_parse_learning_curve_best_epoch(regression_reference):
    """Best epoch from learning-curve.out must stay 182 (Al reference)."""
    expected_epoch = regression_reference['training']['best_epoch']
    expected_rmse = regression_reference['training']['best_test_rmse']

    lc_path = FIXTURES_AL / regression_reference['training']['learning_curve_file']
    with lc_path.open(encoding='utf-8') as handle:
        best_epoch = nnpTrainParser.parseLearningCurve(handle)

    assert best_epoch == expected_epoch, (
        f'Regression failure: expected best epoch {expected_epoch} '
        f'(Al dump-949), got {best_epoch}. '
        'If you changed the parser intentionally, update REFERENCE.json.'
    )

    # Cross-check RMSE at that epoch
    import numpy as np

    data = np.genfromtxt(
        lc_path,
        comments='#',
        usecols=(0, 2),
        dtype=[('epoch', int), ('value', float)],
    )
    rmse = float(data['value'][data['epoch'] == best_epoch][0])
    assert rmse == pytest.approx(expected_rmse), (
        f'Regression failure: RMSE at epoch {best_epoch} was {rmse}, '
        f'expected {expected_rmse}.'
    )
