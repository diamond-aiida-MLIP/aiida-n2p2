"""Run the n2p2 pipeline for Boron on HPC (MakeNNPWorkchain).

Network settings (epochs, hidden layers, nodes, activation) are taken from
``input.nn`` in this directory and can be overridden without editing the file::

    export N2P2_TARGET_EPOCHS=5000
    export N2P2_HIDDEN_LAYERS=2
    export N2P2_NODES='25 25'
    export N2P2_ACTIVATION='p p l'

For a local smoke test::

    export N2P2_COMPUTER=localhost
    python aiida-n2p2_demo.py

Override codes explicitly if needed::

    export N2P2_SCALE_CODE='n2p2_scale@localhost'
    export N2P2_TRAIN_CODE='n2p2_train@localhost'
    export N2P2_LAMMPS_CODE='lammps@localhost'
"""

from __future__ import annotations

import os
from pathlib import Path

from aiida import load_profile, orm
from aiida.engine import run_get_node
from aiida.orm import Bool, Dict, Int, SinglefileData
from aiida.plugins import DataFactory, WorkflowFactory

load_profile()

INPUT_DIR = Path(
    os.environ.get(
        'N2P2_INPUT_DIR',
        Path(__file__).resolve().parent,
    )
)
COMPUTER = os.environ.get('N2P2_COMPUTER', 'dahu_parallel')
SCALE_CODE = os.environ.get('N2P2_SCALE_CODE', f'n2p2_scale@{COMPUTER}')
TRAIN_CODE = os.environ.get('N2P2_TRAIN_CODE', f'n2p2_train@{COMPUTER}')
LAMMPS_CODE = os.environ.get('N2P2_LAMMPS_CODE', f'lammps@{COMPUTER}')
OAR_PROJECT = os.environ.get('N2P2_OAR_PROJECT', 'pr-diamond')
SCALE_NBIN = int(os.environ.get('N2P2_SCALE_NBIN', '100'))
TRAIN_MPI = int(os.environ.get('N2P2_TRAIN_MPI', '4'))
WALLTIME = int(os.environ.get('N2P2_WALLTIME_SECONDS', '3600'))
RUN_VALIDATION = os.environ.get('N2P2_RUN_VALIDATION', 'true').lower() in (
    '1',
    'true',
    'yes',
)

N2p2Dataset = DataFactory('n2p2.dataset')
N2p2Parameters = DataFactory('n2p2.parameters')


def _build_nn_overrides() -> dict:
    """Map optional environment variables to ``input.nn`` keyword overrides."""
    overrides: dict[str, str] = {}
    if 'N2P2_TARGET_EPOCHS' in os.environ:
        overrides['epochs'] = os.environ['N2P2_TARGET_EPOCHS']
    if 'N2P2_HIDDEN_LAYERS' in os.environ:
        overrides['global_hidden_layers_short'] = os.environ['N2P2_HIDDEN_LAYERS']
    if 'N2P2_NODES' in os.environ:
        overrides['global_nodes_short'] = os.environ['N2P2_NODES']
    if 'N2P2_ACTIVATION' in os.environ:
        overrides['global_activation_short'] = os.environ['N2P2_ACTIVATION']
    return overrides


def _effective_keywords(parameters, overrides: dict) -> dict:
    preview = parameters.clone()
    if overrides:
        preview.apply_overrides(overrides)
    return preview.get_attribute('keywords')


computer = orm.load_computer(COMPUTER)
scale_code = orm.load_code(SCALE_CODE)
train_code = orm.load_code(TRAIN_CODE)
lammps_code = orm.load_code(LAMMPS_CODE)

print(f'Computer: {computer.label} ({computer.scheduler_type})')
print(f'Scale code: {scale_code.full_label}')
print(f'Train code: {train_code.full_label}')
print(f'LAMMPS code: {lammps_code.full_label}')
if COMPUTER != 'localhost':
    print(f'OAR project: {OAR_PROJECT}')
print(f'Input directory: {INPUT_DIR}')

dataset = N2p2Dataset.from_file(INPUT_DIR / 'input.data', label='Boron dataset')
parameters = N2p2Parameters.from_file(INPUT_DIR / 'input.nn', label='Boron nn')
nn_overrides = _build_nn_overrides()
keywords = _effective_keywords(parameters, nn_overrides)

print('Network settings (template + overrides):')
print(f"  epochs: {keywords['epochs']['value']}")
print(f"  global_hidden_layers_short: {keywords['global_hidden_layers_short']['value']}")
print(f"  global_nodes_short: {keywords['global_nodes_short']['value']}")
print(f"  global_activation_short: {keywords['global_activation_short']['value']}")
if nn_overrides:
    print(f'  overrides from env: {nn_overrides}')

dataset.store()
parameters.store()

nbin = Int(SCALE_NBIN)
atomic_number = Int(5)

lammps_script = SinglefileData(file=INPUT_DIR / 'in.lmp')
lammps_structure = SinglefileData(file=INPUT_DIR / '222_IN.data')

use_mpi = COMPUTER != 'localhost'
default_metadata = {
    'options': {
        'resources': {
            'num_machines': 1,
            'num_mpiprocs_per_machine': TRAIN_MPI if use_mpi else 1,
        },
        'withmpi': use_mpi,
        'max_wallclock_seconds': WALLTIME,
    }
}
if COMPUTER != 'localhost':
    default_metadata['options']['account'] = OAR_PROJECT

make_nnp = WorkflowFactory('n2p2.make_potential')

prepare_inputs = {
    'dataset': dataset,
    'parameters': parameters,
}
if nn_overrides:
    prepare_inputs['overrides'] = Dict(dict=nn_overrides)

inputs = {
    'prepare': prepare_inputs,
    'scaling': {
        'code': scale_code,
        'nbin': nbin,
        'metadata': default_metadata,
    },
    'training': {
        'code': train_code,
        'atomicNumber': atomic_number,
        'metadata': default_metadata,
    },
    'validation': {
        'code': lammps_code,
        'script': lammps_script,
        'structure': lammps_structure,
        'metadata': default_metadata,
        'retrieve_trajectory': Bool(False),
    },
    'run_validation': Bool(RUN_VALIDATION),
}

print('Running MakeNNPWorkchain (prepare → scale → train → LAMMPS)...')
result, node = run_get_node(make_nnp, **inputs)
print(f'Finished MakeNNPWorkchain<{node.pk}> status={node.exit_status}')
if not node.is_finished_ok:
    raise SystemExit(node.exit_message)
print('Pipeline completed successfully.')
