"""WorkChain for n2p2 training with restart and merged learning curves."""

from __future__ import annotations

import io

from aiida.engine import ToContext, WorkChain, while_
from aiida.orm import Bool, Dict, Int, SinglefileData, WorkChainNode

from aiida_n2p2.calculations.train import nnpTraining
from aiida_n2p2.data.parameters import N2p2Parameters
from aiida_n2p2.utils.learning_curve import (
    build_run_from_curve,
    merge_learning_curve_runs,
    plot_data_to_runs,
    render_merged_learning_curve,
)
from aiida_n2p2.utils.training_state import (
    epoch_offset_for_segment,
    global_last_epoch,
    is_restartable_incomplete,
    local_epochs_from_curve_content,
    resolve_target_epochs,
    training_outputs_available,
    training_reached_target,
)
from aiida_n2p2.workflows.common import CalcJobMetadataWorkChainMixin, exposed_calcjob_inputs


class N2p2TrainWorkChain(CalcJobMetadataWorkChainMixin, WorkChain):
    """Run ``nnp-train``, support restart, and merge learning-curve segments."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(
            nnpTraining,
            exclude=('is_restart', 'restart_weights', 'run_label'),
        )
        spec.input(
            'previous_session',
            valid_type=WorkChainNode,
            required=False,
            non_db=True,
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
            help='Extend ``epochs`` when regenerating ``input.nn`` for manual restart.',
        )
        spec.input(
            'auto_restart',
            valid_type=Bool,
            required=False,
            default=lambda: Bool(False),
            help='Automatically restart until the target ``epochs`` value is reached.',
        )
        spec.input(
            'max_auto_restarts',
            valid_type=Int,
            required=False,
            default=lambda: Int(10),
            help='Maximum number of automatic restart segments.',
        )

        spec.outline(
            cls.initialize_session,
            while_(cls.should_continue_training)(
                cls.setup_training,
                cls.inspect_training,
            ),
            cls.finalize_outputs,
        )

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
        spec.exit_code(
            413,
            'ERROR_MAX_AUTO_RESTARTS',
            message='Training remained incomplete after the maximum number of automatic restarts.',
        )

    def initialize_session(self):
        if (
            'is_restart' in self.inputs
            and self.inputs.is_restart.value
            and 'previous_session' not in self.inputs
        ):
            return self.exit_codes.ERROR_MISSING_RESTART_SOURCE

        self.ctx.run_label = 0
        self.ctx.auto_restart_count = 0
        self.ctx.continue_training = True
        self.ctx.first_segment = True
        self.ctx.merged_plot_data = None
        self.ctx.last_weights = None

        if 'parameters' in self.inputs:
            self.ctx.parameters_template = self.inputs.parameters.clone()
        elif 'auto_restart' in self.inputs and self.inputs.auto_restart.value:
            from aiida_n2p2.data.parameters import N2p2Parameters
            from aiida_n2p2.utils.input_nn import lines_to_dicts, parse_input_nn

            lines = parse_input_nn(io.StringIO(self.inputs.inputNN.get_content()))
            self.ctx.parameters_template = N2p2Parameters.from_lines(
                lines_to_dicts(lines)
            )
        else:
            self.ctx.parameters_template = None

        if 'previous_session' in self.inputs:
            previous = self.inputs.previous_session
            self.ctx.run_label = previous.outputs.session_run_index.value
            self.ctx.merged_plot_data = (
                previous.outputs.learning_curve_plot_data.get_dict()
            )
            if (
                'is_restart' in self.inputs
                and self.inputs.is_restart.value
            ):
                self.ctx.last_weights = previous.outputs.last_weights
        else:
            self.ctx.merged_plot_data = None

        self.ctx.target_epochs = resolve_target_epochs(
            parameters=self.ctx.parameters_template,
            input_nn=self.inputs.inputNN,
        )
        self.ctx.max_auto_restarts = (
            self.inputs.max_auto_restarts.value
            if 'max_auto_restarts' in self.inputs
            else 10
        )

    def should_continue_training(self):
        return self.ctx.continue_training

    def setup_training(self):
        inputs = exposed_calcjob_inputs(
            self,
            nnpTraining,
            exclude=('is_restart', 'restart_weights', 'run_label'),
        )

        self.ctx.run_label += 1
        run_label = self.ctx.run_label
        restart_segment = run_label > 1 or (
            'is_restart' in self.inputs
            and self.inputs.is_restart.value
            and 'previous_session' in self.inputs
        )

        if restart_segment:
            if self.ctx.last_weights is None:
                return self.exit_codes.ERROR_MISSING_RESTART_SOURCE
            inputs['is_restart'] = Bool(True)
            inputs['restart_weights'] = self.ctx.last_weights
            completed_epochs = 0
            if self.ctx.merged_plot_data and self.ctx.merged_plot_data.get('epochs_global'):
                completed_epochs = int(self.ctx.merged_plot_data['epochs_global'][-1])
            input_nn = self._render_restart_input_nn(completed_epochs=completed_epochs)
            if input_nn is not None:
                inputs['inputNN'] = input_nn
        else:
            inputs['is_restart'] = Bool(False)

        inputs['run_label'] = Int(run_label)
        self.ctx.first_segment = False
        self.report(f'Submitting n2p2 training calculation (run {run_label}).')
        future = self.submit(nnpTraining, **inputs)
        return ToContext(training_calc=future)

    def _render_restart_input_nn(self, completed_epochs: int = 0):
        if self.ctx.parameters_template is None:
            return None

        params = self.ctx.parameters_template.clone()
        auto_restart = (
            'auto_restart' in self.inputs and self.inputs.auto_restart.value
        )

        if auto_restart and completed_epochs > 0:
            remaining = max(self.ctx.target_epochs - completed_epochs, 1)
            params.set('epochs', str(remaining))
            params.prepare_for_restart()
            return params.get_singlefile()

        if self.ctx.first_segment and 'additional_epochs' in self.inputs:
            remaining = self.inputs.additional_epochs.value
            params.prepare_for_restart()
            params.set('epochs', str(remaining))
            return params.get_singlefile()

        params.prepare_for_restart(additional_epochs=None)
        return params.get_singlefile()

    def inspect_training(self):
        calc = self.ctx.training_calc
        if not training_outputs_available(calc):
            self.report(
                f'Training failed without recoverable outputs '
                f'(status {calc.exit_status}: {calc.exit_message}).'
            )
            self.ctx.continue_training = False
            return self.exit_codes.ERROR_SUBPROCESS_FAILED

        summary = calc.outputs.training_summary.get_dict()
        previous_runs = (
            plot_data_to_runs(self.ctx.merged_plot_data)
            if self.ctx.merged_plot_data
            else []
        )
        previous_global_max = (
            int(self.ctx.merged_plot_data['epochs_global'][-1])
            if self.ctx.merged_plot_data and self.ctx.merged_plot_data['epochs_global']
            else None
        )
        curve_content = calc.outputs.learning_curve.get_content()
        local_epochs = local_epochs_from_curve_content(curve_content)
        epoch_offset = epoch_offset_for_segment(previous_global_max, local_epochs)

        current_run = build_run_from_curve(
            io.BytesIO(curve_content.encode('utf-8')),
            run_index=self.ctx.run_label - 1,
            epoch_offset=epoch_offset,
            label=f'run-{self.ctx.run_label}'
            + (' (restart)' if self.ctx.run_label > 1 else ''),
        )
        merged = merge_learning_curve_runs(previous_runs + [current_run])
        self.ctx.merged_plot_data = merged
        self.ctx.last_weights = calc.outputs.last_weights
        self.ctx.last_calc = calc

        global_last = global_last_epoch(merged, summary)
        target = self.ctx.target_epochs
        complete = training_reached_target(
            summary=summary,
            merged=merged,
            target_epochs=target,
        )

        if complete:
            self.report(
                f'Training run {self.ctx.run_label} reached target epochs '
                f'({global_last}/{target}; segment local last epoch '
                f'{summary["last_epoch"]}).'
            )
            self.ctx.continue_training = False
            return

        auto_restart = (
            'auto_restart' in self.inputs and self.inputs.auto_restart.value
        )
        if auto_restart and is_restartable_incomplete(
            calc,
            target_epochs=target,
            merged=merged,
        ):
            self.ctx.auto_restart_count += 1
            if self.ctx.auto_restart_count >= self.ctx.max_auto_restarts:
                self.report(
                    f'Training incomplete at epoch {global_last}/{target} after '
                    f'{self.ctx.auto_restart_count} automatic restart(s).'
                )
                self.ctx.continue_training = False
                return self.exit_codes.ERROR_MAX_AUTO_RESTARTS

            self.report(
                f'Training incomplete at epoch {global_last}/{target} '
                f'(segment local last epoch {summary["last_epoch"]}, '
                f'input.nn epochs keyword {summary["target_epochs"]}, '
                f'calc status {calc.exit_status}). '
                f'Automatic restart {self.ctx.auto_restart_count}/'
                f'{self.ctx.max_auto_restarts}.'
            )
            self.ctx.continue_training = True
            return

        if not calc.is_finished_ok:
            self.report(
                f'Training stopped early at epoch {global_last}/{target} '
                f'without automatic restart enabled.'
            )
            self.ctx.continue_training = False
            return self.exit_codes.ERROR_SUBPROCESS_FAILED

        self.report(
            f'Training run {self.ctx.run_label} finished at epoch '
            f'{global_last}/{target}.'
        )
        self.ctx.continue_training = False

    def finalize_outputs(self):
        calc = self.ctx.last_calc
        merged = self.ctx.merged_plot_data
        merged_text = render_merged_learning_curve(merged)
        calc_summary = calc.outputs.training_summary.get_dict()
        global_last = global_last_epoch(merged, calc_summary)

        self.report(
            f'Finalizing training after {self.ctx.run_label} run(s). '
            f'Merged curve spans {len(merged["epochs_global"])} epochs.'
        )

        summary = dict(calc_summary)
        summary['target_epochs'] = self.ctx.target_epochs
        summary['last_epoch'] = global_last
        summary['training_completed'] = global_last >= self.ctx.target_epochs
        summary['epochs_remaining'] = max(self.ctx.target_epochs - global_last, 0)
        summary['n_training_segments'] = self.ctx.run_label

        merged_file = SinglefileData(file=io.BytesIO(merged_text.encode('utf-8')))
        merged_file.store()
        plot_data = Dict(dict=merged)
        plot_data.store()
        summary_node = Dict(dict=summary)
        summary_node.store()
        session_index = Int(self.ctx.run_label)
        session_index.store()

        self.out('weights', calc.outputs.weights)
        self.out('last_weights', calc.outputs.last_weights)
        self.out('learning_curve', calc.outputs.learning_curve)
        self.out('training_summary', summary_node)
        self.out('learning_curve_merged', merged_file)
        self.out('learning_curve_plot_data', plot_data)
        self.out('session_run_index', session_index)
