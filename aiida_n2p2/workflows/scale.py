"""WorkChain for the n2p2 scaling step."""

from aiida.engine import ToContext, WorkChain
from aiida.orm import SinglefileData

from aiida_n2p2.calculations.scaling import nnpScaling
from aiida_n2p2.workflows.common import CalcJobMetadataWorkChainMixin


class N2p2ScaleWorkChain(CalcJobMetadataWorkChainMixin, WorkChain):
    """Run ``nnp-scaling`` and expose the resulting ``scaling.data`` file."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(nnpScaling)
        spec.output(
            'scale',
            valid_type=SinglefileData,
            help='Scaling data produced by ``nnp-scaling``.',
        )
        spec.outline(cls.run_scaling, cls.inspect_scaling)

        spec.exit_code(
            401,
            'ERROR_SUBPROCESS_FAILED',
            message='The scaling CalcJob did not finish successfully.',
        )

    def run_scaling(self):
        self.report('Submitting n2p2 scaling calculation.')
        inputs = self.exposed_inputs(nnpScaling)
        future = self.submit(nnpScaling, **inputs)
        return ToContext(scaling_calc=future)

    def inspect_scaling(self):
        calc = self.ctx.scaling_calc
        if not calc.is_finished_ok:
            self.report(
                f'Scaling failed with status {calc.exit_status}: '
                f'{calc.exit_message}'
            )
            return self.exit_codes.ERROR_SUBPROCESS_FAILED

        self.report('Scaling finished successfully.')
        self.out('scale', calc.outputs.scale)
