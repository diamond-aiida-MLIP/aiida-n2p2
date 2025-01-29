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
            raise exceptions.ParsingError("Can only parse n2p2Train")

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

        # find the best weights
        output_filename = "weights.013.000200.out"

        # add the correct weight file
        self.logger.info(f"Parsing '{output_filename}'")
        with self.retrieved.open(output_filename, "rb") as handle:
            output_node = SinglefileData(file=handle)
        self.out("weights", output_node)

        return ExitCode(0)

    @staticmethod
    def parseLearningCurve(learningcurveFile):
        """_summary_

        Returns:
            _type_: _description_
        """

        # Read the first two columns of the file
        data = np.genfromtxt(
            learningcurveFile,
            comments="#",
            dtype=[("epoch", int), ("value", float)],
        )

        # Find the index of the minimum value in the second column
        min_index = np.argmin(data["value"])

        # Get the corresponding value from the first column
        bestEpoch = data["epoch"][min_index]

        return bestEpoch
