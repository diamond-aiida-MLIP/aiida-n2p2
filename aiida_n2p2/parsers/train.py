"""Parser for n2p2 training CalcJobs."""

from __future__ import annotations

import io
import os

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import Dict, SinglefileData
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

from aiida_n2p2.utils.learning_curve import (
    best_epoch_from_curve,
    last_epoch_from_curve,
)

n2p2Train = CalculationFactory('n2p2.train')


class nnpTrainParser(Parser):
    """Parse training outputs and select best/last weight files."""

    def __init__(self, node):
        super().__init__(node)
        if not issubclass(node.process_class, n2p2Train):
            raise exceptions.ParsingError('Can only parse nnpTraining')

    def parse(self, **kwargs):
        inputs = self.node.inputs
        atomic_number = inputs.atomicNumber.value
        run_label = inputs.run_label.value if hasattr(inputs, 'run_label') else 1

        try:
            with self.retrieved.open('learning-curve.out', 'r') as handle:
                best_epoch, best_test_rmse = best_epoch_from_curve(handle)
            with self.retrieved.open('learning-curve.out', 'rb') as handle:
                learning_curve = SinglefileData(file=handle)
            with self.retrieved.open('learning-curve.out', 'r') as handle:
                last_epoch = last_epoch_from_curve(handle)
        except Exception as exc:
            self.logger.error(f'Failed to parse learning curve: {exc}')
            return ExitCode(401, f'Failed to parse learning curve: {exc}')

        try:
            retrieved_temporary_folder = kwargs['retrieved_temporary_folder']
        except KeyError:
            self.logger.error('Missing retrieved_temporary_folder for weight files')
            return ExitCode(402, 'Missing temporary retrieved folder for weight files.')

        best_weight_file = f'weights.{atomic_number:03d}.{best_epoch:06d}.out'
        last_weight_file = f'weights.{atomic_number:03d}.{last_epoch:06d}.out'

        best_path = os.path.join(retrieved_temporary_folder, best_weight_file)
        last_path = os.path.join(retrieved_temporary_folder, last_weight_file)

        if not os.path.isfile(best_path):
            self.logger.error(f"Best weight file '{best_weight_file}' not found.")
            return ExitCode(
                402,
                f"Best weight file '{best_weight_file}' not found.",
            )
        if not os.path.isfile(last_path):
            self.logger.error(f"Last weight file '{last_weight_file}' not found.")
            return ExitCode(
                402,
                f"Last weight file '{last_weight_file}' not found.",
            )

        with open(best_path, 'rb') as handle:
            self.out('weights', SinglefileData(file=handle))
        with open(last_path, 'rb') as handle:
            self.out('last_weights', SinglefileData(file=handle))
        self.out('learning_curve', learning_curve)

        summary = {
            'atomic_number': atomic_number,
            'run_label': run_label,
            'best_epoch': best_epoch,
            'best_test_rmse': best_test_rmse,
            'last_epoch': last_epoch,
            'best_weights_file': best_weight_file,
            'last_weights_file': last_weight_file,
            'is_restart': bool(
                hasattr(inputs, 'is_restart') and inputs.is_restart.value
            ),
        }
        self.out('training_summary', Dict(dict=summary))

        return ExitCode(0)
