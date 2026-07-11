"""Run the full n2p2 pipeline on HPC for the Al example."""

from pathlib import Path

from aiida import load_profile, orm
from aiida.engine import submit
from aiida.orm import Bool, Dict, Int, SinglefileData
from aiida.plugins import WorkflowFactory

load_profile()

INPUT_DIR = Path(__file__).resolve().parent

computer = orm.load_computer('dahu_parallel')
scale_code = orm.load_code('n2p2_scale@dahu_parallel')
train_code = orm.load_code('n2p2_train@dahu_parallel')
lammps_code = orm.load_code('lammps@dahu_parallel')

nbin = Int(100)
input_data = SinglefileData(file=INPUT_DIR / 'input.data')
input_nn = SinglefileData(file=INPUT_DIR / 'input.nn')
atomic_number = Int(13)

lammps_script = SinglefileData(file=INPUT_DIR / 'in.lmp')
lammps_structure = SinglefileData(file=INPUT_DIR / '222_IN.data')

default_metadata = Dict(
    dict={
        'options': {
            'resources': {'num_machines': 1},
            'max_wallclock_seconds': 3600,
            'account': 'pr-diamond',
        }
    }
)

make_nnp = WorkflowFactory('n2p2.make_potential')

inputs = {
    'scaling': {
        'code': scale_code,
        'nbin': nbin,
        'inputData': input_data,
        'inputNN': input_nn,
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
    },
    'run_validation': Bool(True),
}

node = submit(make_nnp, **inputs, wait=True, wait_interval=60)
print(f'Finished MakeNNPWorkchain<{node.pk}>')
