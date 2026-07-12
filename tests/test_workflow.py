"""Plugin registration and WorkChain specification tests."""

from aiida.plugins import CalculationFactory, ParserFactory, WorkflowFactory


def test_calculation_entry_points_load():
    assert CalculationFactory('n2p2.scale').__name__ == 'nnpScaling'
    assert CalculationFactory('n2p2.train').__name__ == 'nnpTraining'
    assert CalculationFactory('n2p2.predict').__name__ == 'nnpPredict'


def test_parser_entry_points_load():
    assert ParserFactory('n2p2.scale').__name__ == 'nnpScaleParser'
    assert ParserFactory('n2p2.train').__name__ == 'nnpTrainParser'
    assert ParserFactory('n2p2.predict').__name__ == 'nnpPredictParser'


def test_workflow_entry_points_load():
    assert WorkflowFactory('n2p2.scale').__name__ == 'N2p2ScaleWorkChain'
    assert WorkflowFactory('n2p2.train').__name__ == 'N2p2TrainWorkChain'
    assert WorkflowFactory('n2p2.validate_lammps').__name__ == 'N2p2LammpsValidationWorkChain'
    assert WorkflowFactory('n2p2.prepare_inputs').__name__ == 'N2p2PrepareInputsWorkChain'
    assert WorkflowFactory('n2p2.make_potential').__name__ == 'MakeNNPWorkchain'


def test_scale_workchain_exposes_scaling_ports():
    from aiida_n2p2.workflows.scale import N2p2ScaleWorkChain

    spec = N2p2ScaleWorkChain.spec()
    for port in ('code', 'nbin', 'inputData', 'inputNN'):
        assert port in spec.inputs, f'Missing scaling input port: {port}'
    assert 'scale' in spec.outputs


def test_train_workchain_exposes_training_ports():
    from aiida_n2p2.workflows.train import N2p2TrainWorkChain

    spec = N2p2TrainWorkChain.spec()
    for port in ('code', 'atomicNumber', 'inputData', 'inputNN', 'inputScale'):
        assert port in spec.inputs, f'Missing training input port: {port}'
    for port in ('weights', 'last_weights', 'learning_curve_merged'):
        assert port in spec.outputs


def test_validate_lammps_workchain_ports():
    from aiida_n2p2.workflows.validate_lammps import N2p2LammpsValidationWorkChain

    spec = N2p2LammpsValidationWorkChain.spec()
    for port in (
        'code',
        'script',
        'structure',
        'input_nn',
        'scale',
        'weights',
        'atomic_number',
    ):
        assert port in spec.inputs, f'Missing validation input port: {port}'
    assert spec.inputs['metadata'].required is False
    assert 'retrieve_trajectory' in spec.inputs
    assert spec.inputs['retrieve_trajectory'].default().value is False


def test_make_nnp_workchain_spec():
    from aiida_n2p2.workflows.make_potential import MakeNNPWorkchain

    spec = MakeNNPWorkchain.spec()
    scaling = spec.inputs['scaling']
    training = spec.inputs['training']
    validation = spec.inputs['validation']

    for port in ('code', 'nbin', 'inputData', 'inputNN'):
        assert port in scaling, f'Missing scaling.{port}'

    for port in ('code', 'atomicNumber'):
        assert port in training, f'Missing training.{port}'

    for port in ('code', 'script', 'structure'):
        assert port in validation, f'Missing validation.{port}'

    assert 'run_validation' in spec.inputs
    assert 'potential' in spec.outputs
    assert 'scale' in spec.outputs
    assert 'learning_curve_merged' in spec.outputs
    assert 'learning_curve_plot_data' in spec.outputs


def test_prepare_inputs_workchain_ports():
    from aiida_n2p2.workflows.prepare_inputs import N2p2PrepareInputsWorkChain

    spec = N2p2PrepareInputsWorkChain.spec()
    assert 'inputData' in spec.outputs
    assert 'inputNN' in spec.outputs
    assert spec.inputs['dataset'].required is False
    assert spec.inputs['parameters'].required is False


def test_train_workchain_exposes_merged_outputs():
    from aiida_n2p2.workflows.train import N2p2TrainWorkChain

    spec = N2p2TrainWorkChain.spec()
    for port in (
        'learning_curve_merged',
        'learning_curve_plot_data',
        'last_weights',
        'session_run_index',
    ):
        assert port in spec.outputs, f'Missing training output port: {port}'
    assert 'previous_session' in spec.inputs
    assert 'is_restart' in spec.inputs
