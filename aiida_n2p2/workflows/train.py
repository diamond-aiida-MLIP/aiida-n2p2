"""WorkChain for the n2p2 training step."""

from aiida.engine import ToContext, WorkChain

from aiida_n2p2.calculations.train import nnpTraining


class N2p2TrainWorkChain(WorkChain):
    """Run ``nnp-train`` and expose the selected best weights."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(nnpTraining)
        spec.expose_outputs(nnpTraining)
        spec.outline(cls.run_training, cls.inspect_training)

        spec.exit_code(
            402,
            'ERROR_SUBPROCESS_FAILED',
            message='The training CalcJob did not finish successfully.',
        )

    def run_training(self):
        self.report('Submitting n2p2 training calculation.')
        inputs = self.exposed_inputs(nnpTraining)
        future = self.submit(nnpTraining, **inputs)
        return ToContext(training_calc=future)

    def inspect_training(self):
        calc = self.ctx.training_calc
        if not calc.is_finished_ok:
            self.report(
                f'Training failed with status {calc.exit_status}: '
                f'{calc.exit_message}'
            )
            return self.exit_codes.ERROR_SUBPROCESS_FAILED

        self.report('Training finished successfully.')
        self.out('weights', calc.outputs.weights)
