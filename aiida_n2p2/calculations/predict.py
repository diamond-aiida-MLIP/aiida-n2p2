"""AiiDA calculation plugin for nnp-predict."""

from aiida.common import datastructures
from aiida.common.folders import Folder
from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.orm import Code, Int, SinglefileData


class nnpPredict(CalcJob):
    """Run ``nnp-predict`` on a validation dataset."""

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1,
            'num_mpiprocs_per_machine': 1,
        }
        spec.inputs['metadata']['options']['parser_name'].default = 'n2p2.predict'
        spec.input(
            'metadata.options.output_filename',
            valid_type=str,
            default='predict.log',
        )

        spec.input('code', valid_type=Code, help='Executable for nnp-predict')
        spec.input('inputData', valid_type=SinglefileData, help='Validation set')
        spec.input('inputNN', valid_type=SinglefileData, help='Network config')
        spec.input('inputScale', valid_type=SinglefileData, help='Scaling data')
        spec.input(
            'weights',
            valid_type=SinglefileData,
            help='Weights file produced by nnp-train',
        )
        spec.input(
            'param',
            valid_type=Int,
            default=lambda: Int(0),
            help='Write structure information to structure.out (0/1)',
        )

        spec.output('energy', valid_type=SinglefileData)
        spec.output('force', valid_type=SinglefileData)
        spec.output('nnatoms', valid_type=SinglefileData)
        spec.output('config', valid_type=SinglefileData)

        spec.exit_code(
            300,
            'ERROR_MISSING_OUTPUT_FILES',
            message='Calculation did not produce all expected output files.',
        )

    def prepare_for_submission(self, folder: Folder) -> datastructures.CalcInfo:
        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params = [self.inputs.param.value]
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.metadata.options.output_filename

        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = [
            (
                self.inputs.inputData.uuid,
                self.inputs.inputData.filename,
                self.inputs.inputData.filename,
            ),
            (
                self.inputs.inputNN.uuid,
                self.inputs.inputNN.filename,
                self.inputs.inputNN.filename,
            ),
            (
                self.inputs.inputScale.uuid,
                self.inputs.inputScale.filename,
                'scaling.data',
            ),
            (
                self.inputs.weights.uuid,
                self.inputs.weights.filename,
                self.inputs.weights.filename,
            ),
        ]
        calcinfo.retrieve_list = [
            'energy.out',
            'nnforces.out',
            'nnatoms.out',
            'output.data',
        ]

        return calcinfo
