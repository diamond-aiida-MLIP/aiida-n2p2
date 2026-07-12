"""Parser logic tests without submitting calculations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aiida_n2p2.utils.learning_curve import best_epoch_from_curve

FIXTURES_AL = Path(__file__).parent / 'fixtures' / 'al'


def test_parse_learning_curve_best_epoch(regression_reference):
    """Best epoch from learning-curve.out must stay 182 (Al reference)."""
    expected_epoch = regression_reference['training']['best_epoch']
    expected_rmse = regression_reference['training']['best_test_rmse']

    lc_path = FIXTURES_AL / regression_reference['training']['learning_curve_file']
    best_epoch, rmse = best_epoch_from_curve(lc_path)

    assert best_epoch == expected_epoch
    assert rmse == pytest.approx(expected_rmse)

    data = np.genfromtxt(
        lc_path,
        comments='#',
        usecols=(0, 2),
        dtype=[('epoch', int), ('value', float)],
    )
    cross = float(data['value'][data['epoch'] == best_epoch][0])
    assert cross == pytest.approx(expected_rmse)
