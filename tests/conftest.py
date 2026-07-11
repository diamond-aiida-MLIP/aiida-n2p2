"""Shared pytest fixtures for aiida-n2p2 regression tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest_plugins = ['aiida.tools.pytest_fixtures']

FIXTURES_AL = Path(__file__).parent / 'fixtures' / 'al'
REFERENCE_PATH = FIXTURES_AL / 'REFERENCE.json'


def _register_plugin_entry_points() -> None:
    """Register aiida-n2p2 entry points without requiring pip install."""
    from aiida.plugins import entry_point
    from aiida.tools.pytest_fixtures.entry_points import EntryPointManager

    from aiida_n2p2.calculations.predict import nnpPredict
    from aiida_n2p2.calculations.scaling import nnpScaling
    from aiida_n2p2.calculations.train import nnpTraining

    manager = EntryPointManager(entry_point.eps())
    manager.add(nnpScaling, group='aiida.calculations', name='n2p2.scale')
    manager.add(nnpTraining, group='aiida.calculations', name='n2p2.train')
    manager.add(nnpPredict, group='aiida.calculations', name='n2p2.predict')

    entry_point.eps = manager.eps
    entry_point.eps_select = manager.eps_select

    from aiida_n2p2.parsers.predict import nnpPredictParser
    from aiida_n2p2.parsers.scaling import nnpScaleParser
    from aiida_n2p2.parsers.train import nnpTrainParser
    from aiida_n2p2.workflows.make_potential import MakeNNPWorkchain
    from aiida_n2p2.workflows.scale import N2p2ScaleWorkChain
    from aiida_n2p2.workflows.train import N2p2TrainWorkChain
    from aiida_n2p2.workflows.validate_lammps import N2p2LammpsValidationWorkChain

    manager.add(nnpScaleParser, group='aiida.parsers', name='n2p2.scale')
    manager.add(nnpTrainParser, group='aiida.parsers', name='n2p2.train')
    manager.add(nnpPredictParser, group='aiida.parsers', name='n2p2.predict')
    manager.add(N2p2ScaleWorkChain, group='aiida.workflows', name='n2p2.scale')
    manager.add(N2p2TrainWorkChain, group='aiida.workflows', name='n2p2.train')
    manager.add(
        N2p2LammpsValidationWorkChain,
        group='aiida.workflows',
        name='n2p2.validate_lammps',
    )
    manager.add(MakeNNPWorkchain, group='aiida.workflows', name='n2p2.make_potential')


def pytest_configure(config):
    _register_plugin_entry_points()


@pytest.fixture(scope='session')
def regression_reference() -> dict:
    """Load golden reference values (Al, dump-949)."""
    with REFERENCE_PATH.open(encoding='utf-8') as handle:
        return json.load(handle)


@pytest.fixture
def al_fixtures_dir() -> Path:
    return FIXTURES_AL


def _build_retrieved_folder(tmp_path: Path, filenames: list[str]):
    from aiida.orm import FolderData

    retrieved_dir = tmp_path / 'retrieved'
    retrieved_dir.mkdir()
    for name in filenames:
        shutil.copy(FIXTURES_AL / name, retrieved_dir / name)

    folder = FolderData()
    folder.put_object_from_tree(str(retrieved_dir))
    folder.store()
    return folder


@pytest.fixture
def scale_calcjob_node(aiida_profile_clean):
    from aiida.orm import CalcJobNode

    calcjob = CalcJobNode(process_type='aiida.calculations:n2p2.scale')
    calcjob.set_exit_status(0)
    calcjob.store()
    return calcjob


@pytest.fixture
def train_calcjob_node(aiida_profile_clean):
    from aiida.orm import CalcJobNode

    calcjob = CalcJobNode(process_type='aiida.calculations:n2p2.train')
    calcjob.set_exit_status(0)
    calcjob.store()
    return calcjob


@pytest.fixture
def patch_train_atomic_number(monkeypatch, regression_reference):
    """Mock atomicNumber.value without creating extra DB links."""

    def _patch(calcjob):
        inputs = SimpleNamespace(
            atomicNumber=SimpleNamespace(value=regression_reference['atomic_number'])
        )
        monkeypatch.setattr(type(calcjob), 'inputs', property(lambda self, i=inputs: i))

    return _patch


@pytest.fixture
def scale_retrieved_folder(aiida_profile_clean, tmp_path):
    return _build_retrieved_folder(tmp_path, ['scaling.data'])


@pytest.fixture
def train_retrieved_folder(aiida_profile_clean, tmp_path, regression_reference):
    training = regression_reference['training']
    return _build_retrieved_folder(tmp_path, [training['learning_curve_file']])


@pytest.fixture
def train_retrieved_temporary_dir(tmp_path, regression_reference):
    training = regression_reference['training']
    temp_dir = tmp_path / 'retrieved_temporary'
    temp_dir.mkdir()
    shutil.copy(
        FIXTURES_AL / training['best_weights_file'],
        temp_dir / training['best_weights_file'],
    )
    return temp_dir


@pytest.fixture
def patch_parser_retrieved(monkeypatch):
    """Replace Parser.retrieved with a fixture folder."""

    def _patch(parser, folder):
        monkeypatch.setattr(
            type(parser),
            'retrieved',
            property(lambda self, folder=folder: folder),
        )

    return _patch


def pytest_addoption(parser):
    parser.addoption(
        '--regression-report',
        action='store_true',
        default=False,
        help='Print a human-readable regression summary after the test run.',
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not config.getoption('--regression-report'):
        return

    terminalreporter.write_sep('=', 'Regression reference (Al, dump-949)')
    if not REFERENCE_PATH.is_file():
        terminalreporter.write_line('REFERENCE.json not found.')
        return

    with REFERENCE_PATH.open(encoding='utf-8') as handle:
        ref = json.load(handle)

    terminalreporter.write_line(ref['description'])
    terminalreporter.write_line(
        f"  scaling.data md5 ........... {ref['scaling']['md5']}"
    )
    terminalreporter.write_line(
        f"  best epoch ................. {ref['training']['best_epoch']}"
    )
    terminalreporter.write_line(
        f"  best test RMSE ............. {ref['training']['best_test_rmse']}"
    )
    terminalreporter.write_line(
        f"  best weights md5 ........... {ref['training']['best_weights_md5']}"
    )
    terminalreporter.write_line('')
    terminalreporter.write_line(
        'If all tests passed, parsed values still match this reference.'
    )
