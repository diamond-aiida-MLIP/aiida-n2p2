"""
Parser for nnp-scaling functionality in AiiDA-n2p2

The method only parses the scaling.data file and ignores other files.
"""

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import SinglefileData, CalcJobNode
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

n2p2Calculation = CalculationFactory("n2p2.scale")


class nnpScaleParser(Parser):
    """
    Parser class for parsing output of calculation.
    """

    def __init__(self, node: CalcJobNode):
        """Initialize the parser and check if node is compatable

        Args:
            node (CalcJobNode): The calculation node to be parsed

        Raises:
            exceptions.ParsingError: Check correct parser is used
        """
        super().__init__(node)
        if not issubclass(node.process_class, n2p2Calculation):  # type: ignore[arg-type]
            raise exceptions.ParsingError("Can only parse n2p2Calculation")

    def parse(self, **kwargs) -> ExitCode:
        """Parse outputs produced by n2p2.

        Currently it only parses the scaling.data file.

        Returns:
            ExitCode: non-zero exit code, if parsing fails
        """
        output_filename = "scaling.data"

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()
        files_expected = [output_filename]
        # Note: set(A) <= set(B) checks whether A is a subset of B
        if not set(files_expected) <= set(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # add output file
        self.logger.info(f"Parsing '{output_filename}'")
        with self.retrieved.open(output_filename, "rb") as handle:
            output_node = SinglefileData(file=handle)
        self.out("scale", output_node)

        return ExitCode(0)
