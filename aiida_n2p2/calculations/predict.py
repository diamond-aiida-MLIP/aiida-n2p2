"""
Calculations provided by aiida_n2p2.

Register calculations via the "aiida.calculations" entry point in setup.json.
"""

from aiida.common import datastructures
from aiida.engine import CalcJob
from aiida.orm import SinglefileData,Int


class nnpPredict(CalcJob):
    """_summary_

    Args:
        CalcJob (_type_): _description_
    """
    @classmethod
    def define (cls,spec):
        """_summary_

        Args:
            spec (_type_): _description_
        """
        super().define(spec)

        # set default values for AiiDA options
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        spec.inputs["metadata"]["options"]["parser_name"].default = "n2p2.predict"

         # Stdout
        spec.input(
            "metadata.options.output_filename", valid_type=str, default='predict.log'
        )

        #Inputs
      #  spec.input("atomicNumbers",valid_type=str)
      
        
        spec.input(
            "inputData", valid_type=SinglefileData, help="Validation set"
        )
        spec.input(
            "inputNN", valid_type=SinglefileData, help="NN config"
        )
        spec.input(
            "inputScale", valid_type=SinglefileData, help="Scaling data"
        )
        spec.input(
            "weights", valid_type=SinglefileData, help="Weights_file in weights.atomicNumber.epoch format"
        )
        spec.input(
            "param", valid_type=Int, default=lambda: Int(0), help="Write structure information for debugging to structure.out (0/1)"
        )
        
        spec.output(
            "energy", valid_type=SinglefileData, help="The output files with energies"
        )
        spec.output(
            "force", valid_type=SinglefileData, help="The output files with forces"
        )
        spec.output(
            "nnatoms", valid_type=SinglefileData, help="Contains the atomic energy contributions to the total potential energy"
        )
        spec.output(
            "config", valid_type=SinglefileData, help="Contains the configurations with NNP energy and force predictions inserted"
        )


    def prepare_for_submission(self, folder):
                """
                Create input files.

                :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files
                    needed by the calculation.
                :return: `aiida.common.datastructures.CalcInfo` instance
                """
                # Weights file in specific format
             #   weights_file = 'weights.' + str(self.inputs.atomicNumbers) + '.data'

                codeinfo = datastructures.CodeInfo()
                codeinfo.cmdline_params = [self.inputs.param.value]
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
                    (
                        self.inputs.inputScale.uuid,
                        self.inputs.inputScale.filename,
                        'scaling.data'
                    ),
                    (
                        self.inputs.weights.uuid,
                        self.inputs.weights.filename,
                        self.inputs.weights.filename,
                    )

                ]
                calcinfo.retrieve_list = ['energy.out','nnforces.out','nnatoms.out','output.data']

                return calcinfo