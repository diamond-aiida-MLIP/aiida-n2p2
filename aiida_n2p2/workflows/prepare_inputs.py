"""WorkChain to prepare n2p2 input files from structured data nodes."""

from aiida.engine import WorkChain
from aiida.orm import Bool, Dict, Int, SinglefileData

from aiida_n2p2.data.dataset import N2p2Dataset
from aiida_n2p2.data.parameters import N2p2Parameters


class N2p2PrepareInputsWorkChain(WorkChain):
    """Build ``input.data`` and ``input.nn`` for downstream n2p2 calculations."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input(
            'dataset',
            valid_type=N2p2Dataset,
            required=False,
            help='Structured training-set node.',
        )
        spec.input(
            'parameters',
            valid_type=N2p2Parameters,
            required=False,
            help='Structured ``input.nn`` node.',
        )
        spec.input(
            'input_data',
            valid_type=SinglefileData,
            required=False,
            help='Raw ``input.data`` (legacy path).',
        )
        spec.input(
            'input_nn',
            valid_type=SinglefileData,
            required=False,
            help='Raw ``input.nn`` (legacy path).',
        )
        spec.input(
            'overrides',
            valid_type=Dict,
            required=False,
            help='Keyword overrides and optional ``__enable__`` / ``__disable__`` lists.',
        )
        spec.input(
            'is_restart',
            valid_type=Bool,
            required=False,
            default=lambda: Bool(False),
            help='Prepare ``input.nn`` for training restart.',
        )
        spec.input(
            'additional_epochs',
            valid_type=Int,
            required=False,
            help='Extend ``epochs`` by this amount when ``is_restart`` is True.',
        )

        spec.outline(cls.run_prepare)

        spec.output('inputData', valid_type=SinglefileData)
        spec.output('inputNN', valid_type=SinglefileData)
        spec.output('parameters_snapshot', valid_type=Dict)
        spec.output('dataset_metadata', valid_type=Dict)

        spec.exit_code(
            411,
            'ERROR_MISSING_INPUTS',
            message='Provide dataset/parameters or raw input_data/input_nn.',
        )

    def run_prepare(self):
        if 'dataset' in self.inputs:
            input_data = self.inputs.dataset.get_singlefile()
            dataset_metadata = Dict(dict=self.inputs.dataset.as_dict())
        elif 'input_data' in self.inputs:
            input_data = self.inputs.input_data
            dataset_metadata = Dict(dict={'source': 'SinglefileData'})
        else:
            return self.exit_codes.ERROR_MISSING_INPUTS

        if 'parameters' in self.inputs:
            parameters = self.inputs.parameters.clone()
            if 'overrides' in self.inputs:
                parameters.apply_overrides(self.inputs.overrides)
            if self.inputs.is_restart.value:
                additional = (
                    self.inputs.additional_epochs.value
                    if 'additional_epochs' in self.inputs
                    else None
                )
                parameters.prepare_for_restart(additional_epochs=additional)
            input_nn = parameters.get_singlefile()
            parameters_snapshot = Dict(dict=parameters.as_dict())
        elif 'input_nn' in self.inputs:
            input_nn = self.inputs.input_nn
            parameters_snapshot = Dict(dict={'source': 'SinglefileData'})
        else:
            return self.exit_codes.ERROR_MISSING_INPUTS

        self.report('Prepared n2p2 input files.')
        self.out('inputData', input_data)
        self.out('inputNN', input_nn)
        self.out('parameters_snapshot', parameters_snapshot)
        self.out('dataset_metadata', dataset_metadata)
