"""
Calculations provided by aiida_n2p2.

Register calculations via the "aiida.calculations" entry point in setup.json.
"""

from aiida.common import datastructures
from aiida.engine import CalcJob
from aiida.orm import SinglefileData,Int





class nnpScaling(CalcJob):
    """
    AiiDA calculation plugin wrapping the scaling  function.
    """

    @classmethod
    def define(cls, spec):
        """Define inputs and outputs of the calculation."""
        super().define(spec)

        # set default values for AiiDA options
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        spec.inputs["metadata"]["options"]["parser_name"].default = "n2p2.scale"

        # new ports
        spec.input(
            "metadata.options.output_filename", valid_type=str, default='scale.log'
        )

        spec.input("nbin",valid_type=Int)
      
        
        spec.input(
            "inputData", valid_type=SinglefileData, help="Training set"
        )
        spec.input(
            "inputNN", valid_type=SinglefileData, help="Test set"
        )
        spec.output(
            "scale", valid_type=SinglefileData, help="Scaling data"
        )

        spec.exit_code(
            300,
            "ERROR_MISSING_OUTPUT_FILES",
            message="Calculation did not produce all expected output files.",
        )

    def prepare_for_submission(self, folder):
        """
        Create input files.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files
            needed by the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
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
        calcinfo.retrieve_list = ['scaling.data']

        return calcinfo
