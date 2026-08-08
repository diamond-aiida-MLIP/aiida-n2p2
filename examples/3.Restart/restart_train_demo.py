"""Two-session manual training restart demo for aiida-n2p2.

Scale once, train the first half of the target epochs, then submit a second
WorkChain that restarts from ``last_weights`` and completes the remaining epochs.

Uses the **Aluminium** ``input.data`` and ``input.nn`` from ``examples/2.HPC``.

Default: **80 total epochs** split as 40 + 40 (override with ``N2P2_TARGET_EPOCHS``).

**Important:** this example submits **two** ``N2p2TrainWorkChain`` nodes. If you
only see one WorkChain in ``verdi process list``, segment 2 did not run yet.

Usage::

    cd examples/3.Restart
    verdi daemon start
    python restart_train_demo.py

Resume only segment 2 after a successful run 1 (pk from ``verdi process list``)::

    export N2P2_RESUME_RUN1_PK=17672
    python restart_train_demo.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from aiida import load_profile, orm
from aiida.engine import run_get_node
from aiida.orm import Bool, Int
from aiida.plugins import DataFactory, WorkflowFactory

from aiida_n2p2.utils.plot_learning_curve import plot_merged_learning_curve

load_profile()

INPUT_DIR = Path(__file__).resolve().parent
AL_INPUT_DIR = Path(
    os.environ.get(
        'N2P2_INPUT_DIR',
        INPUT_DIR.parent / '2.HPC',
    )
)
TARGET_EPOCHS = int(os.environ.get('N2P2_TARGET_EPOCHS', '80'))
EPOCHS_FIRST = int(
    os.environ.get('N2P2_EPOCHS_FIRST', str(TARGET_EPOCHS // 2))
)
EPOCHS_RESTART = int(
    os.environ.get('N2P2_EPOCHS_RESTART', str(TARGET_EPOCHS - EPOCHS_FIRST))
)
RESUME_RUN1_PK = os.environ.get('N2P2_RESUME_RUN1_PK')
SCALE_NBIN = int(os.environ.get('N2P2_SCALE_NBIN', '100'))
ATOMIC_NUMBER = 13

SCALE_CODE = os.environ.get('N2P2_SCALE_CODE', 'n2p2_scale@localhost')
TRAIN_CODE = os.environ.get('N2P2_TRAIN_CODE', 'n2p2_train@localhost')

N2p2Dataset = DataFactory('n2p2.dataset')
N2p2Parameters = DataFactory('n2p2.parameters')
ScaleWC = WorkflowFactory('n2p2.scale')
TrainWC = WorkflowFactory('n2p2.train')

scale_code = orm.load_code(SCALE_CODE)
train_code = orm.load_code(TRAIN_CODE)

metadata = {
    'options': {
        'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
        'withmpi': False,
    }
}

print(f'Al inputs: {AL_INPUT_DIR}')
print(f'Target epochs (total): {TARGET_EPOCHS}')
print(f'Segment 1: {EPOCHS_FIRST} epochs, segment 2 (restart): +{EPOCHS_RESTART}')

if RESUME_RUN1_PK:
    run1 = orm.load_node(int(RESUME_RUN1_PK))
    if run1.process_class != TrainWC:
        sys.exit(f'Node {run1.pk} is not an N2p2TrainWorkChain.')
    if not run1.is_finished_ok:
        sys.exit(f'Run 1 node {run1.pk} is not finished OK.')
    summary1 = run1.outputs.training_summary.get_dict()
    print(f'Resuming from run 1 pk={run1.pk} ({summary1["last_epoch"]} epochs done)')
    scaling_inputs = run1.inputs
    parameters = orm.load_node(run1.inputs.parameters.pk)
else:
    dataset = N2p2Dataset.from_file(AL_INPUT_DIR / 'input.data', label='Al HPC')
    parameters = N2p2Parameters.from_file(AL_INPUT_DIR / 'input.nn', label='Al HPC nn')
    parameters.set('epochs', str(EPOCHS_FIRST))
    parameters.store()
    dataset.store()

    _, scaling = run_get_node(
        ScaleWC,
        code=scale_code,
        nbin=Int(SCALE_NBIN),
        inputData=dataset.get_singlefile(),
        inputNN=parameters.get_singlefile(),
        metadata=metadata,
    )
    print(f'Scaling finished: pk={scaling.pk}')

    print(f'Submitting manual restart segment 1 ({EPOCHS_FIRST} epochs)...')
    _, run1 = run_get_node(
        TrainWC,
        code=train_code,
        atomicNumber=Int(ATOMIC_NUMBER),
        inputData=scaling.inputs.inputData,
        inputNN=scaling.inputs.inputNN,
        inputScale=scaling.outputs.scale,
        metadata=metadata,
        parameters=parameters,
    )
    summary1 = run1.outputs.training_summary.get_dict()
    print(
        f'Run 1 finished: pk={run1.pk}, '
        f'last epoch {summary1["last_epoch"]}/{EPOCHS_FIRST}, '
        f'best={summary1["best_epoch"]}'
    )
    print(
        f'>>> Segment 1 done ({summary1["last_epoch"]}/{TARGET_EPOCHS} global). '
        f'Starting segment 2...'
    )

    scaling_inputs = scaling.inputs
    parameters = orm.load_node(run1.inputs.parameters.pk)

parameters_restart = parameters.clone()
parameters_restart.set('epochs', str(TARGET_EPOCHS))
parameters_restart.store()

print(f'Submitting manual restart segment 2 (+{EPOCHS_RESTART} epochs)...')
_, run2 = run_get_node(
    TrainWC,
    code=train_code,
    atomicNumber=Int(ATOMIC_NUMBER),
    inputData=scaling_inputs.inputData,
    inputNN=scaling_inputs.inputNN,
    inputScale=run1.inputs.inputScale,
    metadata=metadata,
    is_restart=Bool(True),
    previous_session=run1,
    parameters=parameters_restart,
    additional_epochs=Int(EPOCHS_RESTART),
)
summary2 = run2.outputs.training_summary.get_dict()
plot_data = run2.outputs.learning_curve_plot_data.get_dict()
global_last = plot_data['epochs_global'][-1]

print(
    f'Run 2 (restart) finished: pk={run2.pk}, '
    f'local last epoch {summary2["last_epoch"]}, '
    f'session_run_index={run2.outputs.session_run_index.value}'
)
print(
    f'Merged curve: {plot_data["n_runs"]} run(s), '
    f'global last epoch {global_last} (target {TARGET_EPOCHS})'
)

assert run2.is_finished_ok, run2.exit_message
assert plot_data['n_runs'] == 2
assert run2.outputs.session_run_index.value == 2
assert global_last >= TARGET_EPOCHS - 1, (
    f'Expected at least {TARGET_EPOCHS - 1} global epochs, got {global_last}'
)

plot_path = INPUT_DIR / 'learning_curve_merged.png'
try:
    plot_merged_learning_curve(plot_data, output_path=plot_path)
    print(f'Wrote {plot_path}')
except ImportError:
    print('matplotlib not installed; skipped learning-curve plot')

print('Manual restart demo completed successfully.')
