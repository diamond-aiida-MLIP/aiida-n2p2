"""
Parsers provided by aiida_n2p2.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""

import numpy as np

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import SinglefileData
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

n2p2Train = CalculationFactory("n2p2.train")


class nnpTrainParser(Parser):
    """
    Parser class for parsing output of calculation.
    """

    def __init__(self, node):
        """
        Initialize Parser instance

        Checks that the ProcessNode being passed was produced by a DiffCalculation.

        :param node: ProcessNode of calculation
        :param type node: :class:`aiida.orm.nodes.process.process.ProcessNode`
        """
        super().__init__(node)
        if not issubclass(node.process_class, n2p2Train):
            raise exceptions.ParsingError('Can only parse nnpTraining')

    def parse(self, **kwargs):
        """
        Parse outputs, store results in database.

        :returns: an exit code, if parsing fails (or nothing if parsing succeeds)
        """

        # Check that folder content is as expected
        # files_retrieved = self.retrieved.list_object_names()
        # files_expected = ["learning-curve.out", "weight*.out"]

        # Check that files expected are received

        # Find the best epoch
        # Parse the learning curve to find the best epoch
        best_epoch = None
        try:
            with self.retrieved.open("learning-curve.out", "r") as handle:
                best_epoch = self.parseLearningCurve(handle)
        except Exception as e:
            self.logger.error(f"Failed to parse learning curve: {e}")
            return ExitCode(401, f"Failed to parse learning curve: {str(e)}")

        # find the best weights
        atomic_number = self.node.inputs.atomicNumber.value
        best_weight_file = f"weights.{atomic_number:03d}.{best_epoch:06d}.out"

        self.logger.info(f"Parsing '{best_weight_file}'")
        with self.retrieved.open(best_weight_file, "rb") as handle:
            output_node = SinglefileData(file=handle)
        self.out("weights", output_node)

        return ExitCode(0)

    @staticmethod
    def parseLearningCurve(learningcurveFile):
        """Function to select the optimal epoch from the learning-curve file.
        The lowest test-set error epoch is chosen

        :param learningcurveFile: n2p2 learning curve file.
        """

        # Read the first two columns of the file
        data = np.genfromtxt(
            learningcurveFile,
            comments="#",
            usecols=(0, 2),
            dtype=[("epoch", int), ("value", float)],
        )

        # Find the index of the minimum value in the second column
        min_index = np.argmin(data["value"])

        # Get the corresponding value from the first column
        bestEpoch = data["epoch"][min_index]

        return bestEpoch
