from aiida.engine import WorkChain, ToContext
from aiida.orm import Int, SinglefileData, Code, Dict

from aiida_n2p2.calculations.scaling import nnpScaling
from aiida_n2p2.calculations.train import nnpTraining
from aiida.plugins import CalculationFactory


class MakeNNPWorkchain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("n2p2.scale.code", valid_type=Code)
        spec.input("n2p2.scale.nbin", valid_type=Int)
        spec.input(
            "n2p2.scale.inputData",
            valid_type=SinglefileData,
            help="Training set",
        )
        spec.input(
            "n2p2.scale.inputNN", valid_type=SinglefileData, help="Test set"
        )
        spec.input(
            "n2p2.scale.metadata",
            valid_type=Dict,
            help="Metadata for scaling step",
        )

        spec.input(
            "n2p2.train.code",
            valid_type=Code,
            help="Code for the training step",
        )
        spec.input(
            "n2p2.train.atomicNumber",
            valid_type=Int,
            help="Atomic number of the element",
        )
        spec.input(
            "n2p2.train.metadata",
            valid_type=Dict,
            help="Metadata for training step",
        )

        spec.input(
            "n2p2.validate.code", valid_type=Code, help="Code for LAMMPS"
        )
        spec.input(
            "n2p2.validate.lammpsScript",
            valid_type=SinglefileData,
            help="Script to run LAMMPS",
        )
        spec.input(
            "n2p2.validate.lammpsData",
            valid_type=SinglefileData,
            help="Input structure if used in lammps script",
        )
        spec.input(
            "n2p2.validate.metadata",
            valid_type=Dict,
            required=False,
            help="Metadata for validation step",
        )
        # The outline for the workflow
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
            "code": self.inputs.n2p2.scale.code,
            "nbin": self.inputs.n2p2.scale.nbin,
            "inputData": self.inputs.n2p2.scale.inputData,
            "inputNN": self.inputs.n2p2.scale.inputNN,
            "metadata": self.inputs.n2p2.scale.metadata.get_dict(),
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
            "code": self.inputs.n2p2.train.code,
            "atomicNumber": self.inputs.n2p2.train.atomicNumber,
            "inputData": self.inputs.n2p2.scale.inputData,
            "inputNN": self.inputs.n2p2.scale.inputNN,
            "inputScale": scaledData,
            "metadata": self.inputs.n2p2.train.metadata.get_dict(),
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
        weights_filename = (
            f"weights.{self.inputs.n2p2.train.atomicNumber.value:03d}.data"
        )

        if not training_calc.is_finished_ok:
            self.report("Training step failed.")
            return self.exit_codes.ERROR_TRAINING_FAILED

        self.report("Training calculation finished successfully.")

        LAMMPSCalculation = CalculationFactory("lammps.raw")

        inputs = {
            "code": self.inputs.n2p2.validate.code,
            "script": self.inputs.n2p2.validate.lammpsScript,
            "files": {
                "data": self.inputs.n2p2.validate.lammpsData,
                "inputnn": self.inputs.n2p2.scale.inputNN,
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
            "settings": Dict(
                dict={
                    "additional_retrieve_list": [("*.lammpstrj", ".", None)],
                }
            ),
            "metadata": self.inputs.n2p2.validate.metadata.get_dict(),
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
