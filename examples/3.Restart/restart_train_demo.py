"""Two-session training restart demo for aiida-n2p2 v0.3.0.

Interactive smoke test: scale once, train for 20 epochs, restart for 20 more
epochs, then inspect merged learning-curve metadata.

Requires n2p2 executables configured in AiiDA (``nnp-scaling``, ``nnp-train``).
Adjust ``SCALE_CODE`` and ``TRAIN_CODE`` below to match your profile.

Usage::

    cd examples/3.Restart
    python restart_train_demo.py
"""

from __future__ import annotations

import os
from pathlib import Path

from aiida import load_profile, orm
from aiida.engine import run_get_node
from aiida.orm import Bool, Dict, Int
from aiida.plugins import DataFactory, WorkflowFactory

from aiida_n2p2.utils.plot_learning_curve import plot_merged_learning_curve

load_profile()

INPUT_DIR = Path(__file__).resolve().parent
EPOCHS_FIRST = 20
EPOCHS_RESTART = 20
ATOMIC_NUMBER = 13

# Override with environment variables if your code labels differ.
SCALE_CODE = os.environ.get('N2P2_SCALE_CODE', 'n2p2@local')
TRAIN_CODE = os.environ.get('N2P2_TRAIN_CODE', 'n2p2@local')

N2p2Dataset = DataFactory('n2p2.dataset')
N2p2Parameters = DataFactory('n2p2.parameters')
ScaleWC = WorkflowFactory('n2p2.scale')
TrainWC = WorkflowFactory('n2p2.train')

scale_code = orm.load_code(SCALE_CODE)
train_code = orm.load_code(TRAIN_CODE)

metadata = Dict(
    dict={
        'options': {
            'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
            'withmpi': False,
        }
    }
)

dataset = N2p2Dataset.from_file(INPUT_DIR / 'input.data', label='Al minimal')
parameters = N2p2Parameters.from_file(INPUT_DIR / 'input.nn', label='Al minimal nn')
parameters.set('epochs', str(EPOCHS_FIRST))
parameters.store()
dataset.store()

print(f'Dataset: {dataset.as_dict()}')
print(f'Initial epochs: {parameters.as_dict()["keywords"]["epochs"]["value"]}')

_, scaling = run_get_node(
    ScaleWC,
    code=scale_code,
    nbin=Int(20),
    inputData=dataset.get_singlefile(),
    inputNN=parameters.get_singlefile(),
    metadata=metadata,
)
print(f'Scaling finished: {scaling.pk}')

_, run1 = run_get_node(
    TrainWC,
    code=train_code,
    atomicNumber=Int(ATOMIC_NUMBER),
    inputData=scaling.inputs.inputData,
    inputNN=scaling.inputs.inputNN,
    inputScale=scaling.outputs.scale,
    metadata=metadata,
)
summary1 = run1.outputs.training_summary.get_dict()
print(
    f'Run 1 finished: pk={run1.pk}, '
    f'epochs 1–{summary1["last_epoch"]}, '
    f'best={summary1["best_epoch"]}'
)

parameters_restart = parameters.clone()
parameters_restart.store()

_, run2 = run_get_node(
    TrainWC,
    code=train_code,
    atomicNumber=Int(ATOMIC_NUMBER),
    inputData=scaling.inputs.inputData,
    inputNN=scaling.inputs.inputNN,
    inputScale=scaling.outputs.scale,
    metadata=metadata,
    is_restart=Bool(True),
    previous_session=run1,
    parameters=parameters_restart,
    additional_epochs=Int(EPOCHS_RESTART),
)
summary2 = run2.outputs.training_summary.get_dict()
plot_data = run2.outputs.learning_curve_plot_data.get_dict()

print(
    f'Run 2 (restart) finished: pk={run2.pk}, '
    f'local epochs through {summary2["last_epoch"]}, '
    f'session_run_index={run2.outputs.session_run_index.value}'
)
print(
    f'Merged curve: {plot_data["n_runs"]} run(s), '
    f'{len(plot_data["epochs_global"])} global epochs'
)
assert plot_data['n_runs'] == 2
assert run2.outputs.session_run_index.value == 2
assert plot_data['epochs_global'][-1] == summary1['last_epoch'] + summary2['last_epoch']

plot_path = INPUT_DIR / 'learning_curve_merged.png'
try:
    plot_merged_learning_curve(plot_data, output_path=plot_path)
    print(f'Wrote {plot_path}')
except ImportError:
    print('matplotlib not installed; skipped learning-curve plot')
