"""
nnp-scale  provided by aiida-n2p2.
"""

from aiida.common import datastructures
from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.common.folders import Folder
from aiida.orm import SinglefileData, Int, Code


class nnpScaling(CalcJob):
    """AiiDA calculation interface for nnp-scaling executable in n2p2 package."""

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """Method to define inputs, outputs and exitcodes.

        Args:
            spec (CalcJobProcessSpec): Spec
        """
        super().define(spec)

        # set default values for AiiDA options
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 8,
        }

        spec.input("metadata.options.parser_name", valid_type=str, default="n2p2.scale")

        spec.input(
            "metadata.options.output_filename",
            valid_type=str,
            default="scale.log",
        )

        spec.input("code", valid_type=Code, help="Executable for nnp-scaling")

        spec.input(
            "nbin",
            valid_type=Int,
            default=lambda: Int(500),
            help="Number of bins for histogram",
        )

        spec.input("inputData", valid_type=SinglefileData, help="Training data set")
        spec.input(
            "inputNN",
            valid_type=SinglefileData,
            help="Neural network architecture and hyper params",
        )
        spec.output("scale", valid_type=SinglefileData, help="File with scaling data")

        spec.exit_code(
            300,
            "ERROR_MISSING_OUTPUT_FILES",
            message="Calculation did not produce all expected output files.",
        )

    def prepare_for_submission(self, folder: Folder) -> datastructures.CalcInfo:
        """Describes  the procedure for execution of `CalcJob`.

        Args:
            folder (Folder): Folder to temporarily write files on disk

        Returns:
            datastructures.CalcInfo: An instance of calcinfo to be passed to execution manager.
        """
        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params = [self.inputs.nbin.value]
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.metadata.options.output_filename

        # Prepare a `CalcInfo` to be returned to the engine
        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = [
            (
                self.inputs.inputData.uuid,
                self.inputs.inputData.filename,
                self.inputs.inputData.filename,
            ),
            (
                self.inputs.inputNN.uuid,
                self.inputs.inputNN.filename,
                self.inputs.inputNN.filename,
            ),
        ]
        calcinfo.retrieve_list = [
            "scaling.data",
            self.metadata.options.output_filename,
        ]

        return calcinfo
