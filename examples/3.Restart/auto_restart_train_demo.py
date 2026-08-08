"""Automatic training restart demo for aiida-n2p2 v0.4.0.

Smoke test for ``auto_restart=True``: one WorkChain submit, multiple CalcJob
segments when ``max_wallclock_seconds`` stops training before ``epochs`` is reached.

**Important:** uses Al inputs from ``examples/2.HPC`` (not the tiny fixture files
in this directory). Walltime limits are enforced by the cluster scheduler (Slurm, PBS,
etc.). On a plain ``local`` AiiDA computer the job may finish in a single segment
and auto-restart will not trigger. Run this example on HPC, or set a very short
walltime on a machine whose scheduler honours ``max_wallclock_seconds``.

Usage::

    cd examples/3.Restart
    export N2P2_TRAIN_CODE='n2p2_train@dahu_parallel'   # if needed
    export N2P2_WALLTIME_SECONDS=120                      # optional, default 120
    python auto_restart_train_demo.py

Success criteria:

- ``node.is_finished_ok``
- ``training_summary['training_completed']`` is ``True``
- ``learning_curve_plot_data['n_runs'] >= 2`` (at least one automatic restart)
"""

from __future__ import annotations

import os
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
WALLTIME_SECONDS = int(os.environ.get('N2P2_WALLTIME_SECONDS', '120'))
MAX_AUTO_RESTARTS = int(os.environ.get('N2P2_MAX_AUTO_RESTARTS', '10'))
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

scale_metadata = {
    'options': {
        'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 4},
        'withmpi': False,
    }
}

train_metadata = {
    'options': {
        'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 4},
        'withmpi': False,
        'max_wallclock_seconds': WALLTIME_SECONDS,
    }
}

dataset = N2p2Dataset.from_file(AL_INPUT_DIR / 'input.data', label='Al HPC')
parameters = N2p2Parameters.from_file(AL_INPUT_DIR / 'input.nn', label='Al auto-restart')
parameters.set('epochs', str(TARGET_EPOCHS))
parameters.store()
dataset.store()

print(f'Al inputs: {AL_INPUT_DIR}')
print(f'Target epochs: {TARGET_EPOCHS}')
print(f'Walltime per segment: {WALLTIME_SECONDS} s')
print(f'Max automatic restarts: {MAX_AUTO_RESTARTS}')

_, scaling = run_get_node(
    ScaleWC,
    code=scale_code,
    nbin=Int(SCALE_NBIN),
    inputData=dataset.get_singlefile(),
    inputNN=parameters.get_singlefile(),
    metadata=scale_metadata,
)
print(f'Scaling finished: {scaling.pk}')

_, node = run_get_node(
    TrainWC,
    code=train_code,
    atomicNumber=Int(ATOMIC_NUMBER),
    inputData=scaling.inputs.inputData,
    inputNN=scaling.inputs.inputNN,
    inputScale=scaling.outputs.scale,
    metadata=train_metadata,
    parameters=parameters,
    auto_restart=Bool(True),
    max_auto_restarts=Int(MAX_AUTO_RESTARTS),
)

summary = node.outputs.training_summary.get_dict()
plot_data = node.outputs.learning_curve_plot_data.get_dict()
n_calcjobs = len(
    [
        child
        for child in node.called_descendants
        if child.process_type.endswith('n2p2.train')
    ]
)

print(f'TrainWorkChain finished: pk={node.pk}, status={node.exit_status}')
print(
    f"Training completed: {summary.get('training_completed')} "
    f"({plot_data['epochs_global'][-1]}/{summary.get('target_epochs')} epochs)"
)
print(f'CalcJob segments: {n_calcjobs}')
print(f'Merged runs: {plot_data["n_runs"]}')

if plot_data['n_runs'] < 2:
    print(
        'WARNING: only one training segment was executed. '
        'This usually means walltime was not enforced (typical on local AiiDA). '
        'Re-run on HPC with a shorter N2P2_WALLTIME_SECONDS or more target epochs.'
    )
else:
    print('Auto-restart loop executed successfully.')

plot_path = INPUT_DIR / 'learning_curve_auto_restart.png'
try:
    plot_merged_learning_curve(plot_data, output_path=plot_path)
    print(f'Wrote {plot_path}')
except ImportError:
    print('matplotlib not installed; skipped plot')

assert node.is_finished_ok, node.exit_message
assert summary.get('training_completed') is True
