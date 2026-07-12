"""WorkChain for n2p2 training with restart and merged learning curves."""

from __future__ import annotations

import io

from aiida.engine import ToContext, WorkChain
from aiida.orm import Bool, Dict, Int, SinglefileData, WorkChainNode

from aiida_n2p2.calculations.train import nnpTraining
from aiida_n2p2.data.parameters import N2p2Parameters
from aiida_n2p2.utils.learning_curve import (
    build_run_from_curve,
    merge_learning_curve_runs,
    plot_data_to_runs,
    render_merged_learning_curve,
)


class N2p2TrainWorkChain(WorkChain):
    """Run ``nnp-train``, support restart, and merge learning-curve segments."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(nnpTraining)
        spec.input(
            'previous_session',
            valid_type=WorkChainNode,
            required=False,
            help='Previous training session used for restart and curve merging.',
        )
        spec.input(
            'is_restart',
            valid_type=Bool,
            required=False,
            default=lambda: Bool(False),
            help='Restart from ``previous_session.outputs.last_weights``.',
        )
        spec.input(
            'parameters',
            valid_type=N2p2Parameters,
            required=False,
            help='Optional ``input.nn`` template; used to enable restart keywords.',
        )
        spec.input(
            'additional_epochs',
            valid_type=Int,
            required=False,
            help='Extend ``epochs`` when regenerating ``input.nn`` for restart.',
        )

        spec.outline(cls.setup_training, cls.inspect_training)

        spec.output('weights', valid_type=SinglefileData)
        spec.output('last_weights', valid_type=SinglefileData)
        spec.output('learning_curve', valid_type=SinglefileData)
        spec.output('training_summary', valid_type=Dict)
        spec.output('learning_curve_merged', valid_type=SinglefileData)
        spec.output('learning_curve_plot_data', valid_type=Dict)
        spec.output('session_run_index', valid_type=Int)

        spec.exit_code(
            402,
            'ERROR_SUBPROCESS_FAILED',
            message='The training CalcJob did not finish successfully.',
        )
        spec.exit_code(
            412,
            'ERROR_MISSING_RESTART_SOURCE',
            message='Restart requested but previous_session was not provided.',
        )

    def setup_training(self):
        inputs = self.exposed_inputs(nnpTraining, exclude=('is_restart', 'restart_weights', 'run_label'))

        if self.inputs.is_restart.value:
            if 'previous_session' not in self.inputs:
                return self.exit_codes.ERROR_MISSING_RESTART_SOURCE
            previous = self.inputs.previous_session
            run_label = previous.outputs.session_run_index.value + 1
            inputs['is_restart'] = Bool(True)
            inputs['restart_weights'] = previous.outputs.last_weights
            if 'parameters' in self.inputs:
                params = self.inputs.parameters.clone()
                additional = (
                    self.inputs.additional_epochs.value
                    if 'additional_epochs' in self.inputs
                    else None
                )
                params.prepare_for_restart(additional_epochs=additional)
                inputs['inputNN'] = params.get_singlefile()
        else:
            run_label = 1
            inputs['is_restart'] = Bool(False)

        inputs['run_label'] = Int(run_label)
        self.ctx.run_label = run_label

        self.report(f'Submitting n2p2 training calculation (run {run_label}).')
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

        run_label = self.ctx.run_label
        run_index = run_label - 1

        if 'previous_session' in self.inputs:
            previous_plot = self.inputs.previous_session.outputs.learning_curve_plot_data.get_dict()
            previous_runs = plot_data_to_runs(previous_plot)
            epoch_offset = previous_plot['epochs_global'][-1] + 1
        else:
            previous_runs = []
            epoch_offset = 0

        current_run = build_run_from_curve(
            io.BytesIO(calc.outputs.learning_curve.get_content().encode('utf-8')),
            run_index=run_index,
            epoch_offset=epoch_offset,
            label=f'run-{run_label}' + (' (restart)' if run_label > 1 else ''),
        )
        merged = merge_learning_curve_runs(previous_runs + [current_run])
        merged_text = render_merged_learning_curve(merged)

        self.report(
            f'Training run {run_label} finished. Merged curve spans '
            f'{len(merged["epochs_global"])} epochs across {merged["n_runs"]} run(s).'
        )

        self.out('weights', calc.outputs.weights)
        self.out('last_weights', calc.outputs.last_weights)
        self.out('learning_curve', calc.outputs.learning_curve)
        self.out('training_summary', calc.outputs.training_summary)
        self.out(
            'learning_curve_merged',
            SinglefileData(file=io.BytesIO(merged_text.encode('utf-8'))),
        )
        self.out('learning_curve_plot_data', Dict(dict=merged))
        self.out('session_run_index', Int(run_label))
