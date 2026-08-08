"""Automatic training restart on Dahu (OAR) for aiida-n2p2 v0.4.0.

Configured for::

    n2p2_scale@dahu_parallel
    n2p2_train@dahu_parallel
    computer: dahu_parallel  (scheduler: oarscheduler)
    project:  pr-diamond

Each training segment gets ``max_wallclock_seconds`` via ``#OAR -l walltime=...``.
The WorkChain resubmits automatically until ``epochs`` is reached.

Run from your laptop (AiiDA daemon must be running and able to reach Dahu)::

    cd examples/3.Restart
    verdi daemon status          # start if needed: verdi daemon start
    python auto_restart_dahu.py

Tune segment length / target epochs with environment variables::

    export N2P2_WALLTIME_SECONDS=60    # 60 s per OAR job (default; 200 epochs need ~113 s)
    export N2P2_TARGET_EPOCHS=200      # total target in input.nn (matches Al file)
    export N2P2_SCALE_MPI=4            # mpirun -n 4 for nnp-scaling
    export N2P2_TRAIN_MPI=4            # mpirun -n 4 for nnp-train
    export N2P2_SCALE_NBIN=500
    export N2P2_MAX_AUTO_RESTARTS=10
    python auto_restart_dahu.py

Success: ``learning_curve_plot_data['n_runs'] >= 2`` and
``training_summary['training_completed'] is True``.
"""

from __future__ import annotations

import os
from pathlib import Path

from aiida import load_profile, orm
from aiida.engine import submit
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
COMPUTER = 'dahu_parallel'
SCALE_CODE_LABEL = 'n2p2_scale@dahu_parallel'
TRAIN_CODE_LABEL = 'n2p2_train@dahu_parallel'
OAR_PROJECT = os.environ.get('N2P2_OAR_PROJECT', 'pr-diamond')

TARGET_EPOCHS = int(os.environ.get('N2P2_TARGET_EPOCHS', '300'))
WALLTIME_SECONDS = int(os.environ.get('N2P2_WALLTIME_SECONDS', '360'))
MAX_AUTO_RESTARTS = int(os.environ.get('N2P2_MAX_AUTO_RESTARTS', '10'))
SCALE_MPI = int(os.environ.get('N2P2_SCALE_MPI', '4'))
TRAIN_MPI = int(os.environ.get('N2P2_TRAIN_MPI', '4'))
SCALE_NBIN = int(os.environ.get('N2P2_SCALE_NBIN', '500'))
ATOMIC_NUMBER = 13

N2p2Dataset = DataFactory('n2p2.dataset')
N2p2Parameters = DataFactory('n2p2.parameters')
ScaleWC = WorkflowFactory('n2p2.scale')
TrainWC = WorkflowFactory('n2p2.train')

computer = orm.load_computer(COMPUTER)
scale_code = orm.load_code(SCALE_CODE_LABEL)
train_code = orm.load_code(TRAIN_CODE_LABEL)

print(f'Computer: {computer.label} ({computer.scheduler_type})')
print(f'Scale code: {scale_code.full_label}')
print(f'Train code: {train_code.full_label}')
print(f'OAR project: {OAR_PROJECT}')
print(f'Target epochs: {TARGET_EPOCHS}')
print(f'Walltime per segment: {WALLTIME_SECONDS} s')
print(f'Al inputs: {AL_INPUT_DIR}')
print(f'Scaling MPI procs: {SCALE_MPI}, nbin: {SCALE_NBIN}')
print(f'Training MPI procs: {TRAIN_MPI}')

hpc_options = {
    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': TRAIN_MPI},
    'withmpi': True,
    'account': OAR_PROJECT,
    'max_wallclock_seconds': WALLTIME_SECONDS,
}

scale_metadata = {
    'options': {
        'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': SCALE_MPI},
        'withmpi': True,
        'account': OAR_PROJECT,
        'max_wallclock_seconds': 1800,
    }
}

train_metadata = {'options': hpc_options}

dataset = N2p2Dataset.from_file(AL_INPUT_DIR / 'input.data', label='Al dahu')
parameters = N2p2Parameters.from_file(AL_INPUT_DIR / 'input.nn', label='Al dahu')
parameters.set('epochs', str(TARGET_EPOCHS))
parameters.store()
dataset.store()

print('Submitting scaling WorkChain...')
scaling = submit(
    ScaleWC,
    code=scale_code,
    nbin=Int(SCALE_NBIN),
    inputData=dataset.get_singlefile(),
    inputNN=parameters.get_singlefile(),
    metadata=scale_metadata,
    wait=True,
    wait_interval=30,
)
if not scaling.is_finished_ok:
    raise SystemExit(f'Scaling failed: {scaling.exit_status} {scaling.exit_message}')
print(f'Scaling finished: pk={scaling.pk}')

print('Submitting training WorkChain with auto_restart=True...')
node = submit(
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
    wait=True,
    wait_interval=60,
)

if not node.is_finished_ok:
    raise SystemExit(f'Training failed: {node.exit_status} {node.exit_message}')

summary = node.outputs.training_summary.get_dict()
plot_data = node.outputs.learning_curve_plot_data.get_dict()
train_calcs = [
    child
    for child in node.called_descendants
    if 'n2p2.train' in child.process_type
]

print(f'TrainWorkChain finished: pk={node.pk}')
print(
    f"Completed: {summary.get('training_completed')} — "
    f"global epochs {plot_data['epochs_global'][-1]}/"
    f"{summary.get('target_epochs')}"
)
print(f'CalcJob segments: {len(train_calcs)}')
print(f'Merged runs: {plot_data["n_runs"]}')
for calc in train_calcs:
    print(f'  segment pk={calc.pk} exit={calc.exit_status} finished_ok={calc.is_finished_ok}')

if plot_data['n_runs'] < 2:
    print(
        'WARNING: only one segment ran. Try lowering N2P2_WALLTIME_SECONDS '
        'or raising N2P2_TARGET_EPOCHS.'
    )
else:
    print('Auto-restart on Dahu: OK')

plot_path = INPUT_DIR / 'learning_curve_auto_restart_dahu.png'
try:
    plot_merged_learning_curve(plot_data, output_path=plot_path)
    print(f'Wrote {plot_path}')
except ImportError:
    print('matplotlib not installed; skipped plot')
