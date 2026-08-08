"""
Parser for nnp-scaling functionality in AiiDA-n2p2

The method only parses the scaling.data file and ignores other files.
"""

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import SinglefileData, CalcJobNode
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

from aiida_n2p2.utils.remote_outputs import fetch_remote_output_file

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
            raise exceptions.ParsingError('Can only parse nnpScaling')

    def parse(self, **kwargs) -> ExitCode:
        """Parse outputs produced by n2p2.

        Currently it only parses the scaling.data file.

        Returns:
            ExitCode: non-zero exit code, if parsing fails
        """
        output_filename = "scaling.data"

        if output_filename in self.retrieved.list_object_names():
            self.logger.info(f"Parsing '{output_filename}'")
            with self.retrieved.open(output_filename, "rb") as handle:
                output_node = SinglefileData(file=handle)
        else:
            self.logger.error(
                "Found files %r, expected to find %r",
                self.retrieved.list_object_names(),
                [output_filename],
            )
            remote_path = fetch_remote_output_file(
                self.node,
                output_filename,
                logger=self.logger,
            )
            if remote_path is None:
                return self.exit_codes.ERROR_MISSING_OUTPUT_FILES
            self.logger.info(f"Parsing '{output_filename}' recovered from remote")
            try:
                with remote_path.open("rb") as handle:
                    output_node = SinglefileData(file=handle)
            finally:
                remote_path.unlink(missing_ok=True)

        self.out("scale", output_node)
        return ExitCode(0)
