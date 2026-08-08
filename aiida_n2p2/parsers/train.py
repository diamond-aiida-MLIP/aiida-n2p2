"""Parser for n2p2 training CalcJobs."""

from __future__ import annotations

import io
import os
import tempfile

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import Dict, SinglefileData
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

from aiida_n2p2.utils.input_nn import target_epochs_from_input_nn
from aiida_n2p2.utils.learning_curve import (
    best_epoch_from_curve,
    last_epoch_from_curve,
)
from aiida_n2p2.utils.remote_outputs import (
    fetch_remote_files_to_directory,
    fetch_stable_remote_bytes,
    remote_weight_epoch_numbers,
)
from aiida_n2p2.utils.training_state import resolve_last_training_epoch

n2p2Train = CalculationFactory('n2p2.train')
LEARNING_CURVE_FILENAME = 'learning-curve.out'


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
            curve_bytes, weight_epochs = self._read_training_progress_bytes(
                atomic_number,
            )
            curve_text = curve_bytes.decode('utf-8')
            best_epoch, best_test_rmse = best_epoch_from_curve(io.StringIO(curve_text))
            curve_last_epoch = last_epoch_from_curve(io.StringIO(curve_text))
            last_epoch, used_weight_epochs = resolve_last_training_epoch(
                curve_last_epoch=curve_last_epoch,
                weight_epochs=weight_epochs,
            )
            if used_weight_epochs:
                self.logger.warning(
                    'Learning curve ends at epoch %s but remote weights exist '
                    'through epoch %s; using weights for restart checkpoint.',
                    curve_last_epoch,
                    last_epoch,
                )
                curve_bytes, _ = self._read_training_progress_bytes(
                    atomic_number,
                    force_remote=True,
                )
                curve_text = curve_bytes.decode('utf-8')
                best_epoch, best_test_rmse = best_epoch_from_curve(io.StringIO(curve_text))
                curve_last_epoch = last_epoch_from_curve(io.StringIO(curve_text))
                last_epoch, _ = resolve_last_training_epoch(
                    curve_last_epoch=curve_last_epoch,
                    weight_epochs=weight_epochs,
                )
            learning_curve = SinglefileData(file=io.BytesIO(curve_bytes))
        except Exception as exc:
            self.logger.error(f'Failed to parse learning curve: {exc}')
            return ExitCode(401, f'Failed to parse learning curve: {exc}')

        temp_folder, cleanup_temp_folder = self._resolve_temporary_folder(kwargs)
        try:
            best_weight_file = f'weights.{atomic_number:03d}.{best_epoch:06d}.out'
            last_weight_file = f'weights.{atomic_number:03d}.{last_epoch:06d}.out'
            if not self._ensure_weight_files(
                temp_folder,
                (best_weight_file, last_weight_file),
            ):
                missing = [
                    name
                    for name in (best_weight_file, last_weight_file)
                    if not os.path.isfile(os.path.join(temp_folder, name))
                ]
                self.logger.error('Weight files not found after remote recovery: %r', missing)
                return ExitCode(
                    402,
                    f'Weight files not found: {", ".join(missing)}',
                )

            best_path = os.path.join(temp_folder, best_weight_file)
            last_path = os.path.join(temp_folder, last_weight_file)

            with open(best_path, 'rb') as handle:
                self.out('weights', SinglefileData(file=handle))
            with open(last_path, 'rb') as handle:
                self.out('last_weights', SinglefileData(file=handle))
            self.out('learning_curve', learning_curve)

            try:
                with self.retrieved.open(inputs.inputNN.filename, 'r') as handle:
                    target_epochs = target_epochs_from_input_nn(handle)
            except Exception:
                target_epochs = target_epochs_from_input_nn(
                    io.StringIO(inputs.inputNN.get_content())
                )
        except Exception as exc:
            self.logger.error(f'Failed to read target epochs from input.nn: {exc}')
            return ExitCode(403, f'Failed to read target epochs from input.nn: {exc}')
        finally:
            if cleanup_temp_folder:
                for name in os.listdir(temp_folder):
                    os.remove(os.path.join(temp_folder, name))
                os.rmdir(temp_folder)

        training_completed = last_epoch >= target_epochs

        summary = {
            'atomic_number': atomic_number,
            'run_label': run_label,
            'best_epoch': best_epoch,
            'best_test_rmse': best_test_rmse,
            'last_epoch': last_epoch,
            'curve_last_epoch': curve_last_epoch,
            'weight_last_epoch': weight_epochs[-1] if weight_epochs else None,
            'target_epochs': target_epochs,
            'training_completed': training_completed,
            'epochs_remaining': max(target_epochs - last_epoch, 0),
            'best_weights_file': best_weight_file,
            'last_weights_file': last_weight_file,
            'is_restart': bool(
                hasattr(inputs, 'is_restart') and inputs.is_restart.value
            ),
        }
        self.out('training_summary', Dict(dict=summary))

        if training_completed:
            return ExitCode(0)

        self.logger.warning(
            'Training stopped before reaching target epochs '
            f'({last_epoch}/{target_epochs}). Outputs were parsed for restart.'
        )
        return ExitCode(0)

    def _read_training_progress_bytes(
        self,
        atomic_number: int,
        *,
        force_remote: bool = False,
    ) -> tuple[bytes, list[int]]:
        retrieved_bytes = None
        if LEARNING_CURVE_FILENAME in self.retrieved.list_object_names():
            with self.retrieved.open(LEARNING_CURVE_FILENAME, 'rb') as handle:
                retrieved_bytes = handle.read()

        remote_bytes = None
        if force_remote or 'remote_folder' in self.node.outputs:
            remote_bytes = fetch_stable_remote_bytes(
                self.node,
                LEARNING_CURVE_FILENAME,
                logger=self.logger,
            )

        if remote_bytes is not None:
            if retrieved_bytes is not None and len(remote_bytes) > len(retrieved_bytes):
                self.logger.warning(
                    'Using remote %r (%s bytes) instead of premature retrieved copy (%s bytes).',
                    LEARNING_CURVE_FILENAME,
                    len(remote_bytes),
                    len(retrieved_bytes),
                )
            curve_bytes = remote_bytes
        elif retrieved_bytes is not None:
            curve_bytes = retrieved_bytes
        else:
            raise FileNotFoundError(LEARNING_CURVE_FILENAME)

        weight_epochs = remote_weight_epoch_numbers(self.node, atomic_number)
        return curve_bytes, weight_epochs

    def _resolve_temporary_folder(self, kwargs: dict) -> tuple[str, bool]:
        try:
            return kwargs['retrieved_temporary_folder'], False
        except KeyError:
            self.logger.warning(
                'Missing retrieved_temporary_folder; creating a local temporary folder.'
            )
            return tempfile.mkdtemp(), True

    def _ensure_weight_files(
        self,
        folder: str,
        filenames: tuple[str, ...],
    ) -> bool:
        missing = [
            name
            for name in filenames
            if not os.path.isfile(os.path.join(folder, name))
        ]
        if not missing:
            return True

        self.logger.error(
            'Weight files %r missing from retrieved temporary folder %r.',
            missing,
            folder,
        )
        return fetch_remote_files_to_directory(
            self.node,
            missing,
            folder,
            initial_delay=0,
            logger=self.logger,
        )
