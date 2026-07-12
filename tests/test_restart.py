"""Tests for training restart behaviour and merged learning curves."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from aiida.plugins import ParserFactory

from aiida_n2p2.data.parameters import N2p2Parameters
from aiida_n2p2.utils.learning_curve import (
    build_run_from_curve,
    last_epoch_from_curve,
    merge_learning_curve_runs,
    plot_data_to_runs,
    render_merged_learning_curve,
)

FIXTURES = Path(__file__).parent / 'fixtures' / 'al'
EXAMPLE_NN = Path(__file__).parent.parent / 'examples' / '1.Al' / 'input.nn'


def test_last_epoch_differs_from_best_epoch(regression_reference):
    lc = FIXTURES / regression_reference['training']['learning_curve_file']
    assert last_epoch_from_curve(lc) == regression_reference['training']['last_epoch']
    assert last_epoch_from_curve(lc) != regression_reference['training']['best_epoch']


def test_train_parser_stores_best_and_last_weights(
    train_calcjob_node,
    train_retrieved_folder,
    train_retrieved_temporary_dir,
    patch_parser_retrieved,
    patch_train_atomic_number,
    regression_reference,
):
    """Parser must expose both best-epoch and last-epoch weight files."""
    patch_train_atomic_number(train_calcjob_node)
    parser = ParserFactory('n2p2.train')(train_calcjob_node)
    patch_parser_retrieved(parser, train_retrieved_folder)
    result = parser.parse(
        retrieved_temporary_folder=str(train_retrieved_temporary_dir)
    )
    assert result.status == 0

    training = regression_reference['training']
    summary = parser.outputs.training_summary.get_dict()
    assert summary['best_epoch'] == training['best_epoch']
    assert summary['last_epoch'] == training['last_epoch']
    assert summary['is_restart'] is False

    assert parser.outputs.weights.filename == training['best_weights_file']
    assert parser.outputs.last_weights.filename == training['last_weights_file']


def test_restart_prepare_generates_weights_data_filename():
    """Restart prep must enable use_old_weights_short in input.nn."""
    params = N2p2Parameters.from_file(EXAMPLE_NN)
    params.prepare_for_restart(additional_epochs=50)
    rendered = params.render()
    assert 'use_old_weights_short' in rendered
    assert 'epochs' in rendered


def test_merged_learning_curve_restart_session(regression_reference):
    """Simulate two training runs and verify merge offsets and run colours."""
    lc = FIXTURES / regression_reference['training']['learning_curve_file']

    run1 = build_run_from_curve(
        lc,
        run_index=0,
        epoch_offset=0,
        label='run-1',
        color='#1f77b4',
    )
    run2 = build_run_from_curve(
        lc,
        run_index=1,
        epoch_offset=int(run1.epochs_local[-1]) + 1,
        label='run-2 (restart)',
        color='#ff7f0e',
    )
    merged = merge_learning_curve_runs([run1, run2])

    assert merged['n_runs'] == 2
    assert merged['runs'][0]['color'] == '#1f77b4'
    assert merged['runs'][1]['color'] == '#ff7f0e'
    assert merged['runs'][1]['epoch_offset'] == 201
    assert merged['epochs_global'][201] == 201
    assert merged['run_index'][200] == 0
    assert merged['run_index'][201] == 1

    merged_text = render_merged_learning_curve(merged)
    assert 'run-2 (restart)' in merged_text
    assert 'global_epoch run_index local_epoch' in merged_text

    rebuilt = plot_data_to_runs(merged)
    assert len(rebuilt) == 2
    assert rebuilt[1].label == 'run-2 (restart)'


def test_simulated_session_merge_matches_workchain_logic(regression_reference):
    """Mirror N2p2TrainWorkChain.inspect_training merge for a restart."""
    lc = FIXTURES / regression_reference['training']['learning_curve_file']

    first_plot = merge_learning_curve_runs(
        [build_run_from_curve(lc, run_index=0, epoch_offset=0, label='run-1')]
    )
    previous_runs = plot_data_to_runs(first_plot)
    epoch_offset = first_plot['epochs_global'][-1] + 1

    current_run = build_run_from_curve(
        io.BytesIO(Path(lc).read_text(encoding='utf-8').encode('utf-8')),
        run_index=1,
        epoch_offset=epoch_offset,
        label='run-2 (restart)',
    )
    merged = merge_learning_curve_runs(previous_runs + [current_run])

    assert merged['epochs_global'][-1] == first_plot['epochs_global'][-1] + current_run.n_epochs
    assert merged['run_index'].count(0) == first_plot['epochs_global'][-1] + 1


def test_train_calcjob_restart_stages_weights_data_file(
    tmp_path, n2p2_train_builder_inputs
):
    """Restart CalcJob must stage checkpoint weights as weights.ZZZ.data."""
    from aiida.common.folders import Folder
    from aiida.engine import ProcessBuilder
    from aiida.orm import Bool
    from aiida.plugins import CalculationFactory

    nnp_training = CalculationFactory('n2p2.train')
    builder = ProcessBuilder(nnp_training)
    for key, value in n2p2_train_builder_inputs.items():
        setattr(builder, key, value)
    builder.is_restart = Bool(True)
    builder.metadata.options.resources = {
        'num_machines': 1,
        'num_mpiprocs_per_machine': 1,
    }

    process = nnp_training(inputs=builder._inputs(prune=True))
    folder = Folder(str(tmp_path / 'sandbox'))
    calcinfo = process.prepare_for_submission(folder)

    destinations = [entry[2] for entry in calcinfo.local_copy_list]
    assert 'weights.013.data' in destinations
    restart_entries = [
        entry
        for entry in calcinfo.local_copy_list
        if entry[2] == 'weights.013.data'
    ]
    assert len(restart_entries) == 1
    assert (
        restart_entries[0][0]
        == n2p2_train_builder_inputs['restart_weights'].uuid
    )


def test_train_workchain_restart_wires_calcjob_submit(
    monkeypatch,
    n2p2_train_builder_inputs,
    finished_train_session_node,
):
    """N2p2TrainWorkChain restart must forward checkpoint weights to the CalcJob."""
    from types import MethodType, SimpleNamespace
    from unittest.mock import MagicMock

    from aiida.orm import Bool, Dict
    from aiida.plugins import CalculationFactory

    from aiida_n2p2.workflows.train import N2p2TrainWorkChain

    class Inputs(SimpleNamespace):
        def __contains__(self, key):
            return hasattr(self, key)

    captured: dict = {}

    def intercept_submit(_self, process_class, **inputs):
        captured['process_class'] = process_class
        captured['inputs'] = inputs
        return MagicMock()

    def fake_exposed_inputs(_self, _cls, exclude=()):
        excluded = set(exclude)
        return {
            key: value
            for key, value in workchain.inputs.__dict__.items()
            if not key.startswith('_') and key not in excluded
        }

    monkeypatch.setattr(N2p2TrainWorkChain, 'submit', intercept_submit)
    monkeypatch.setattr(
        'aiida_n2p2.workflows.train.ToContext',
        lambda *args, **kwargs: kwargs,
    )

    inputs_obj = Inputs(
        code=n2p2_train_builder_inputs['code'],
        atomicNumber=n2p2_train_builder_inputs['atomicNumber'],
        inputData=n2p2_train_builder_inputs['inputData'],
        inputNN=n2p2_train_builder_inputs['inputNN'],
        inputScale=n2p2_train_builder_inputs['inputScale'],
        metadata=Dict(
            dict={
                'options': {
                    'resources': {
                        'num_machines': 1,
                        'num_mpiprocs_per_machine': 1,
                    }
                }
            }
        ),
        is_restart=Bool(True),
        previous_session=finished_train_session_node,
    )

    workchain = object.__new__(N2p2TrainWorkChain)
    workchain.report = lambda _message: None
    workchain.exposed_inputs = MethodType(fake_exposed_inputs, workchain)
    workchain._parsed_inputs = inputs_obj
    workchain._context = SimpleNamespace()

    workchain.setup_training()

    assert captured['process_class'] is CalculationFactory('n2p2.train')
    inputs = captured['inputs']
    assert inputs['is_restart'].value is True
    assert inputs['run_label'].value == 2
    assert (
        inputs['restart_weights'].uuid
        == n2p2_train_builder_inputs['restart_weights'].uuid
    )
