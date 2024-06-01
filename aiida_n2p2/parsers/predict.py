from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import SinglefileData
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

n2p2Calculation = CalculationFactory("n2p2.predict")


class nnpPredictParser(Parser):
    """
    Parser class for parsing output of calculation.
    """

    def __init__(self, node):
        """
        Initialize Parser instance

        Checks that the ProcessNode being passed was produced by a n2p2Calculation.

        :param node: ProcessNode of calculation
        :param type node: :class:`aiida.orm.nodes.process.process.ProcessNode`
        """
        super().__init__(node)
        if not issubclass(node.process_class, n2p2Calculation):
            raise exceptions.ParsingError("Can only parse n2p2Calculation")

    def parse(self, **kwargs):
        """
        Parse outputs, store results in database.

        :returns: an exit code, if parsing fails (or nothing if parsing succeeds)
        """
        # Define the expected output filenames
        outFilenames = {
            'energy': 'energy.out',
            'force': 'nnforces.out',
            'nnatoms': 'nnatoms.out',
            'config': 'output.data'
        }

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()
        files_expected = list(outFilenames.values())
        # Note: set(A) <= set(B) checks whether A is a subset of B
        if not set(files_expected) <= set(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # add output file
        for label, outFilename in outFilenames.items():
            with self.retrieved.open(outFilename, "rb") as handle:
                output_node = SinglefileData(file=handle)
                self.out(label, output_node)

        return ExitCode(0)
