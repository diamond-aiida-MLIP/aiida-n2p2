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


def test_workflow_entry_point_load():
    workchain = WorkflowFactory('n2p2.make_potential')
    assert workchain.__name__ == 'MakeNNPWorkchain'


def test_make_nnp_workchain_spec():
    """Input/output ports must stay stable for downstream workflows."""
    from aiida_n2p2.workflows.make_potential import MakeNNPWorkchain

    spec = MakeNNPWorkchain.spec()
    scale = spec.inputs['n2p2']['scale']
    train = spec.inputs['n2p2']['train']
    validate = spec.inputs['n2p2']['validate']

    for port in ('code', 'nbin', 'inputData', 'inputNN', 'metadata'):
        assert port in scale, f'Missing n2p2.scale.{port}'

    for port in ('code', 'atomicNumber', 'metadata'):
        assert port in train, f'Missing n2p2.train.{port}'

    for port in ('code', 'lammpsScript', 'lammpsData'):
        assert port in validate, f'Missing n2p2.validate.{port}'

    assert 'potential' in spec.outputs
    assert 'scale' in spec.outputs
    assert spec.outputs['potential'].valid_type.__name__ == 'SinglefileData'
    assert spec.outputs['scale'].valid_type.__name__ == 'SinglefileData'
