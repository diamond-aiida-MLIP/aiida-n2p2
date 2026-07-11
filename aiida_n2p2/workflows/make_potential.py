"""End-to-end workchain composing scaling, training, and LAMMPS validation."""

from aiida.engine import ToContext, WorkChain, if_
from aiida.orm import Bool, SinglefileData

from aiida_n2p2.workflows.scale import N2p2ScaleWorkChain
from aiida_n2p2.workflows.train import N2p2TrainWorkChain
from aiida_n2p2.workflows.validate_lammps import N2p2LammpsValidationWorkChain


class MakeNNPWorkchain(WorkChain):
    """Build an n2p2 potential through scaling, training, and optional LAMMPS validation."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(N2p2ScaleWorkChain, namespace='scaling')
        spec.expose_inputs(
            N2p2TrainWorkChain,
            namespace='training',
            exclude=('inputData', 'inputNN', 'inputScale'),
        )
        spec.expose_inputs(
            N2p2LammpsValidationWorkChain,
            namespace='validation',
            exclude=('input_nn', 'scale', 'weights', 'atomic_number'),
        )
        spec.input(
            'run_validation',
            valid_type=Bool,
            required=False,
            default=lambda: Bool(True),
            help='If False, skip the LAMMPS validation step.',
        )

        spec.outline(
            cls.run_scaling,
            cls.run_training,
            if_(cls.should_run_validation)(cls.run_validation),
            cls.finalize,
        )

        spec.output('potential', valid_type=SinglefileData)
        spec.output('scale', valid_type=SinglefileData)

        spec.exit_code(
            201,
            'ERROR_SCALING_FAILED',
            message='Scaling step failed.',
        )
        spec.exit_code(
            202,
            'ERROR_TRAINING_FAILED',
            message='Training step failed.',
        )
        spec.exit_code(
            203,
            'ERROR_VALIDATION_FAILED',
            message='LAMMPS validation step failed.',
        )

    def should_run_validation(self):
        return self.inputs.run_validation.value

    def run_scaling(self):
        self.report('Launching scaling work chain.')
        inputs = self.exposed_inputs(N2p2ScaleWorkChain, namespace='scaling')
        future = self.submit(N2p2ScaleWorkChain, **inputs)
        return ToContext(scaling_work=future)

    def run_training(self):
        scaling_work = self.ctx.scaling_work
        if not scaling_work.is_finished_ok:
            self.report('Scaling work chain failed.')
            return self.exit_codes.ERROR_SCALING_FAILED

        self.report('Launching training work chain.')
        training_inputs = self.exposed_inputs(
            N2p2TrainWorkChain,
            namespace='training',
        )
        training_inputs['inputData'] = scaling_work.inputs.inputData
        training_inputs['inputNN'] = scaling_work.inputs.inputNN
        training_inputs['inputScale'] = scaling_work.outputs.scale

        future = self.submit(N2p2TrainWorkChain, **training_inputs)
        return ToContext(training_work=future)

    def run_validation(self):
        training_work = self.ctx.training_work
        scaling_work = self.ctx.scaling_work

        if not training_work.is_finished_ok:
            self.report('Training work chain failed.')
            return self.exit_codes.ERROR_TRAINING_FAILED

        self.report('Launching LAMMPS validation work chain.')
        validation_inputs = self.exposed_inputs(
            N2p2LammpsValidationWorkChain,
            namespace='validation',
        )
        validation_inputs['input_nn'] = scaling_work.inputs.inputNN
        validation_inputs['scale'] = scaling_work.outputs.scale
        validation_inputs['weights'] = training_work.outputs.weights
        validation_inputs['atomic_number'] = training_work.inputs.atomicNumber

        future = self.submit(N2p2LammpsValidationWorkChain, **validation_inputs)
        return ToContext(validation_work=future)

    def finalize(self):
        training_work = self.ctx.training_work
        scaling_work = self.ctx.scaling_work

        if not training_work.is_finished_ok:
            self.report('Training work chain failed.')
            return self.exit_codes.ERROR_TRAINING_FAILED

        if self.inputs.run_validation.value:
            validation_work = self.ctx.validation_work
            if not validation_work.is_finished_ok:
                self.report('LAMMPS validation work chain failed.')
                return self.exit_codes.ERROR_VALIDATION_FAILED

        self.report('Workflow finished successfully.')
        self.out('potential', training_work.outputs.weights)
        self.out('scale', scaling_work.outputs.scale)
