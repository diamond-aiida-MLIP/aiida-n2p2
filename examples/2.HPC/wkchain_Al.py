

import subprocess
import time

from pathlib import Path

from aiida.engine import run, submit, run_get_node
from aiida import orm
from aiida.orm import Int, SinglefileData, Code, load_node
from aiida.plugins import WorkflowFactory
from aiida.manage.configuration import load_profile

from aiida.common.exceptions import NotExistent

load_profile()

INPUT_DIR = Path(__file__).resolve().parent
print(INPUT_DIR)

# Create or load code
computer = orm.load_computer('DU')
scaleCode = orm.load_code('n2p2_scale@DU')
trainCode= orm.load_code('n2p2_train@DU')
lammpsCode=orm.load_code('lammps@DU')
#computer = orm.load_computer('JZ')
#scaleCode = orm.load_code('scale@JZ')
#trainCode= orm.load_code('train@JZ')
#lammpsCode=orm.load_code('lammps@JZ')


# Parameters for n2p2 training
nbin = orm.Int(100)
inputData = orm.SinglefileData(file=INPUT_DIR / 'input.data')
inputNN = orm.SinglefileData(file=INPUT_DIR / 'input.nn')
atomicNumber=orm.Int(13)

#LAMMPS validation input
lammpsScript=orm.SinglefileData(file=INPUT_DIR / 'in.lmp')
lammpsData=orm.SinglefileData(file=INPUT_DIR / '222_IN.data')

#Define resources
scale_metadata = Dict(dict={
    "options": {
        "resources": {"num_machines": 1},
        "max_wallclock_seconds": 3600
    }
})

train_metadata = Dict(dict={
    "options": {
        "resources": {"num_machines": 1},
        "max_wallclock_seconds": 3600
    }
})

validate_metadata = Dict(dict={
    "options": {
        "resources": {"num_machines": 1},
        "max_wallclock_seconds": 3600
    }
})


train_n2p2=WorkflowFactory('n2p2.make_potential')

# Run the WorkChain
inputs = {
    "n2p2": {
        "scale": {
            "code": scaleCode,
            "nbin": nbin,
            "inputData": inputData,
            "inputNN": inputNN,
            "metadata": scale_metadata,
        },
        "train": {
            "code": trainCode,
            "atomicNumber": atomicNumber,
            "metadata": train_metadata,
        },
        "validate": {
            "code": lammpsCode,
            "lammpsScript": lammpsScript,
            "lammpsData": lammpsData,
            "metadata": validate_metadata,
        }
    }
}

# Run the WorkChain
node=submit(train_n2p2,**inputs,wait=True,wait_interval=60)
pk=node.pk

# Dump outputs in local folder
print(f" Dumping process details for PK {pk}")
subprocess.run(["verdi", "process", "dump", str(node.pk)])
