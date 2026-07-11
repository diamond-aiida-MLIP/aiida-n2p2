"""AiiDA calculation plugin for nnp-train."""

from aiida.common import datastructures
from aiida.common.folders import Folder
from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.orm import Code, Int, SinglefileData


class nnpTraining(CalcJob):
    """Run ``nnp-train`` to optimise neural-network weights."""

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        spec.inputs['metadata']['options']['resources'].default = {
            'num_machines': 1,
            'num_mpiprocs_per_machine': 8,
        }
        spec.inputs['metadata']['options']['parser_name'].default = 'n2p2.train'
        spec.input(
            'metadata.options.output_filename',
            valid_type=str,
            default='train.log',
        )

        spec.input('code', valid_type=Code, help='Executable for nnp-train')
        spec.input(
            'atomicNumber',
            valid_type=Int,
            help='Atomic number of the element being trained',
        )
        spec.input('inputData', valid_type=SinglefileData, help='Training set')
        spec.input(
            'inputNN',
            valid_type=SinglefileData,
            help='Neural network architecture and hyperparameters',
        )
        spec.input('inputScale', valid_type=SinglefileData, help='Scaling data')
        spec.output(
            'weights',
            valid_type=SinglefileData,
            help='Best weights selected from the learning curve',
        )
        spec.exit_code(
            300,
            'ERROR_MISSING_OUTPUT_FILES',
            message='Calculation did not produce all expected output files.',
        )

    def prepare_for_submission(self, folder: Folder) -> datastructures.CalcInfo:
        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params = []
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
        ]
        # Keep only the learning curve in the repository; epoch weight files are
        # retrieved temporarily and discarded after the parser selects the best one.
        calcinfo.retrieve_list = ['learning-curve.out']
        calcinfo.retrieve_temporary_list = ['weights.*.out']

        return calcinfo
