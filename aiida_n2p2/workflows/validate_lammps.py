"""WorkChain for LAMMPS validation of a trained n2p2 potential."""

from aiida.engine import ToContext, WorkChain
from aiida.orm import Bool, Dict, Int, SinglefileData, Code
from aiida.plugins import CalculationFactory

from aiida_n2p2.workflows.common import metadata_dict


class N2p2LammpsValidationWorkChain(WorkChain):
    """Validate a trained potential with a short LAMMPS run."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('code', valid_type=Code, help='LAMMPS executable')
        spec.input(
            'script',
            valid_type=SinglefileData,
            help='LAMMPS input script',
        )
        spec.input(
            'structure',
            valid_type=SinglefileData,
            help='Structure file used by the LAMMPS script',
        )
        spec.input(
            'input_nn',
            valid_type=SinglefileData,
            help='n2p2 input.nn file copied into the LAMMPS run folder',
        )
        spec.input(
            'scale',
            valid_type=SinglefileData,
            help='Scaling data copied into the LAMMPS run folder',
        )
        spec.input(
            'weights',
            valid_type=SinglefileData,
            help='Best weights file produced by n2p2 training',
        )
        spec.input(
            'atomic_number',
            valid_type=Int,
            help='Atomic number used to rename the weights file for LAMMPS',
        )
        spec.input(
            'metadata',
            valid_type=Dict,
            required=False,
            help='Optional scheduler metadata for the LAMMPS CalcJob',
        )
        spec.input(
            'retrieve_trajectory',
            valid_type=Bool,
            required=False,
            default=lambda: Bool(False),
            help=(
                'If True, retrieve LAMMPS trajectory files (*.lammpstrj). '
                'These can be very large for long MD runs.'
            ),
        )

        spec.outline(cls.run_validation, cls.inspect_validation)

        spec.exit_code(
            403,
            'ERROR_SUBPROCESS_FAILED',
            message='The LAMMPS validation CalcJob did not finish successfully.',
        )

    def run_validation(self):
        self.report('Submitting LAMMPS validation calculation.')
        lammps = CalculationFactory('lammps.raw')
        weights_filename = f'weights.{self.inputs.atomic_number.value:03d}.data'

        settings = {}
        if self.inputs.retrieve_trajectory.value:
            settings['additional_retrieve_list'] = [('*.lammpstrj', '.', None)]

        inputs = {
            'code': self.inputs.code,
            'script': self.inputs.script,
            'files': {
                'data': self.inputs.structure,
                'inputnn': self.inputs.input_nn,
                'scale': self.inputs.scale,
                'weight': self.inputs.weights,
            },
            'filenames': Dict(
                dict={
                    'data': 'IN.data',
                    'inputnn': 'input.nn',
                    'scale': 'scaling.data',
                    'weight': weights_filename,
                }
            ),
            'metadata': metadata_dict(
                self.inputs.metadata if 'metadata' in self.inputs else None
            ),
        }
        if settings:
            inputs['settings'] = Dict(dict=settings)
        future = self.submit(lammps, **inputs)
        return ToContext(validation_calc=future)

    def inspect_validation(self):
        calc = self.ctx.validation_calc
        if not calc.is_finished_ok:
            self.report(
                f'LAMMPS validation failed with status {calc.exit_status}: '
                f'{calc.exit_message}'
            )
            return self.exit_codes.ERROR_SUBPROCESS_FAILED

        self.report('LAMMPS validation finished successfully.')
