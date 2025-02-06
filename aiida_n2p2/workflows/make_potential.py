from aiida.engine import WorkChain, ToContext
from aiida.orm import Int, SinglefileData, Code, Dict

from aiida_n2p2.calculations.scaling import nnpScaling
from aiida_n2p2.calculations.train import nnpTraining
from aiida.plugins import CalculationFactory


class MakeNNPWorkchain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code)
        spec.input("nbin", valid_type=Int)
        spec.input("inputData", valid_type=SinglefileData, help="Training set")
        spec.input("inputNN", valid_type=SinglefileData, help="Test set")
        spec.input(
            "atomicNumber", valid_type=Int, help="Atomic number of the element"
        )
        spec.input(
            "trainCode", valid_type=Code, help="Code for the training step"
        )

        spec.input("lammpsCode", valid_type=Code, help="Code for LAMMPS")
        spec.input(
            "lammpsScript",
            valid_type=SinglefileData,
            help="Script to run LAMMPS",
        )
        spec.input(
            "lammpsData",
            valid_type=SinglefileData,
            help="Input structure if used in lammps script",
        )

        spec.outline(cls.scale, cls.train, cls.validate, cls.get_potential)

        spec.output("potential", valid_type=SinglefileData)
        spec.output("scale", valid_type=SinglefileData)

        # Define exit codes for error handling
        spec.exit_code(
            201, "ERROR_SCALING_FAILED", message="Scaling step failed."
        )
        spec.exit_code(
            202, "ERROR_TRAINING_FAILED", message="Training step failed."
        )
        spec.exit_code(
            203, "ERROR_VALIDATION_FAILED", message="Prediction step failed."
        )

    def scale(self):
        """Step 1: Run the scaling CalcJob."""

        inputs = {
            "code": self.inputs.code,
            "nbin": self.inputs.nbin,
            "inputData": self.inputs.inputData,
            "inputNN": self.inputs.inputNN,
            "metadata": {
                "label": "Scaling Step",
                "options": {
                    "resources": {
                        "num_machines": 1,
                        "num_mpiprocs_per_machine": 8,
                    },
                    "withmpi": True,
                },
            },
        }
        self.report("Submitting scaling calculation...")
        future = self.submit(nnpScaling, **inputs)
        return ToContext(scaling_calc=future)

    def train(self):
        """Step 2: Run the training CalcJob."""
        scaling_calc = self.ctx.scaling_calc
        if not scaling_calc.is_finished_ok:
            self.report("Scaling step failed.")
            return self.exit_codes.ERROR_SCALING_FAILED
        else:
            self.report("Scaling calculation  finished successfully.")

        scaledData = scaling_calc.outputs.scale

        inputs = {
            "code": self.inputs.trainCode,
            "atomicNumber": self.inputs.atomicNumber,
            "inputData": self.inputs.inputData,
            "inputNN": self.inputs.inputNN,
            "inputScale": scaledData,
            "metadata": {
                "label": "Training Step",
                "options": {"withmpi": False},
            },
        }
        self.report("Submitting Training calculation...")
        future = self.submit(nnpTraining, **inputs)
        return ToContext(training_calc=future)

    def validate(self):
        """Step 3: Run a validation test using LAMMPS.
        Problems  there are custom lines in thermo
        """
        training_calc = self.ctx.training_calc
        scaling_calc = self.ctx.scaling_calc
        atomic_number = self.inputs.atomicNumber.value
        weights_filename = f"weights.{atomic_number:3d}.data"

        if not training_calc.is_finished_ok:
            self.report("Training step failed.")
            return self.exit_codes.ERROR_TRAINING_FAILED

        self.report("Training calculation finished successfully.")

        LAMMPSCalculation = CalculationFactory("lammps.raw")

        inputs = {
            "code": self.inputs.lammpsCode,
            "script": self.inputs.lammpsScript,
            "files": {
                "data": self.inputs.lammpsData,
                "inputnn": self.inputs.inputNN,
                "scale": scaling_calc.outputs.scale,
                "weight": training_calc.outputs.weights,
            },
            "filenames": Dict(
                dict={
                    "data": "IN.data",
                    "inputnn": "input.nn",
                    "scale": "scaling.data",
                    "weight": weights_filename,
                }
            ),
            "metadata": {
                "options": {
                    "resources": {
                        "num_machines": 1,
                        "num_mpiprocs_per_machine": 8,
                    },
                    "withmpi": True,
                }
            },
        }
        self.report("Submitting validation calculation using LAMMPS...")
        future = self.submit(LAMMPSCalculation, **inputs)
        return ToContext(validation_calc=future)

    def get_potential(self):
        validation_calc = self.ctx.validation_calc

        if not (validation_calc.is_finished_ok):
            self.report("Validation calculation failed")
            return self.exit_codes.ERROR_VALIDATION_FAILED

        self.report("LAMMPS calculation finished successfully.")
        self.out("potential", self.ctx.training_calc.outputs.weights)
        self.out("scale", self.ctx.scaling_calc.outputs.scale)
