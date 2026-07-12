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

    from aiida_n2p2.data.dataset import N2p2Dataset
    from aiida_n2p2.data.parameters import N2p2Parameters
    from aiida_n2p2.calculations.predict import nnpPredict
    from aiida_n2p2.calculations.scaling import nnpScaling
    from aiida_n2p2.calculations.train import nnpTraining

    manager = EntryPointManager(entry_point.eps())
    manager.add(nnpScaling, group='aiida.calculations', name='n2p2.scale')
    manager.add(nnpTraining, group='aiida.calculations', name='n2p2.train')
    manager.add(nnpPredict, group='aiida.calculations', name='n2p2.predict')
    manager.add(N2p2Dataset, group='aiida.data', name='n2p2.dataset')
    manager.add(N2p2Parameters, group='aiida.data', name='n2p2.parameters')

    entry_point.eps = manager.eps
    entry_point.eps_select = manager.eps_select

    from aiida_n2p2.parsers.predict import nnpPredictParser
    from aiida_n2p2.parsers.scaling import nnpScaleParser
    from aiida_n2p2.parsers.train import nnpTrainParser
    from aiida_n2p2.workflows.make_potential import MakeNNPWorkchain
    from aiida_n2p2.workflows.prepare_inputs import N2p2PrepareInputsWorkChain
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
    manager.add(
        N2p2PrepareInputsWorkChain,
        group='aiida.workflows',
        name='n2p2.prepare_inputs',
    )


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
    for key in ('best_weights_file', 'last_weights_file'):
        shutil.copy(
            FIXTURES_AL / training[key],
            temp_dir / training[key],
        )
    return temp_dir


@pytest.fixture
def n2p2_train_builder_inputs(aiida_profile_clean, aiida_code_installed):
    """Stored inputs for nnpTraining ProcessBuilder tests."""
    from aiida.orm import Int, SinglefileData

    code = aiida_code_installed(
        default_calc_job_plugin='n2p2.train',
        filepath_executable='nnp-train',
    )
    input_data = SinglefileData(file=FIXTURES_AL / 'minimal_input.data').store()
    input_nn = SinglefileData(
        file=Path(__file__).parent.parent / 'examples' / '1.Al' / 'input.nn'
    ).store()
    input_scale = SinglefileData(file=FIXTURES_AL / 'scaling.data').store()
    restart_weights = SinglefileData(
        file=FIXTURES_AL / 'weights.013.000200.out'
    ).store()

    return {
        'code': code,
        'atomicNumber': Int(13),
        'inputData': input_data,
        'inputNN': input_nn,
        'inputScale': input_scale,
        'restart_weights': restart_weights,
    }


@pytest.fixture
def finished_train_session_node(aiida_profile_clean, n2p2_train_builder_inputs):
    """Minimal finished N2p2TrainWorkChain node with restart-relevant outputs."""
    from aiida.common.links import LinkType
    from aiida.orm import Int, WorkChainNode

    node = WorkChainNode(process_type='aiida.workflows:n2p2.train')
    node.store()
    session_index = Int(1).store()
    session_index.base.links.add_incoming(
        node, LinkType.RETURN, 'session_run_index'
    )
    n2p2_train_builder_inputs['restart_weights'].base.links.add_incoming(
        node, LinkType.RETURN, 'last_weights'
    )
    return node


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
