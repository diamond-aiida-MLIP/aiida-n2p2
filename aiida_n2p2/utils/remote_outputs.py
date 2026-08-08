"""Helpers to recover CalcJob outputs from the remote work directory."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Iterable

from aiida.orm import CalcJobNode

DEFAULT_REMOTE_POLL_INITIAL_DELAY = 30
DEFAULT_REMOTE_POLL_INTERVAL = 30
DEFAULT_REMOTE_POLL_MAX_WAIT = 1800
DEFAULT_REMOTE_STABLE_CHECKS = 2


def _poll_setting(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def get_remote_poll_settings(node: CalcJobNode) -> tuple[int, int, int]:
    """Return ``(initial_delay, poll_interval, max_wait)`` in seconds."""
    initial_delay = _poll_setting(
        'N2P2_REMOTE_POLL_INITIAL_DELAY',
        DEFAULT_REMOTE_POLL_INITIAL_DELAY,
    )
    poll_interval = _poll_setting(
        'N2P2_REMOTE_POLL_INTERVAL',
        DEFAULT_REMOTE_POLL_INTERVAL,
    )
    walltime = node.get_option('max_wallclock_seconds')
    if walltime is not None:
        max_wait = int(walltime) + 120
    else:
        max_wait = _poll_setting(
            'N2P2_REMOTE_POLL_MAX_WAIT',
            DEFAULT_REMOTE_POLL_MAX_WAIT,
        )
    return initial_delay, poll_interval, max_wait


def _remote_work_path(node: CalcJobNode) -> str | None:
    if 'remote_folder' not in node.outputs:
        return None
    return node.outputs.remote_folder.get_remote_path()


def remote_weight_epoch_numbers(node: CalcJobNode, atomic_number: int) -> list[int]:
    """Return sorted local epoch numbers available in remote ``weights.*.out`` files."""
    remote_path = _remote_work_path(node)
    if remote_path is None:
        return []

    prefix = f'weights.{atomic_number:03d}.'
    transport = node.get_transport()
    with transport:
        remote_names = transport.listdir(remote_path)

    epochs: list[int] = []
    for name in remote_names:
        if name.startswith(prefix) and name.endswith('.out'):
            epochs.append(int(name.split('.')[-2]))
    return sorted(epochs)


def fetch_stable_remote_bytes(
    node: CalcJobNode,
    filename: str,
    *,
    initial_delay: int | None = None,
    poll_interval: int | None = None,
    max_wait: int | None = None,
    stable_checks: int | None = None,
    logger=None,
) -> bytes | None:
    """Poll a remote file until its size is stable, then return the contents."""
    defaults = get_remote_poll_settings(node)
    if initial_delay is None:
        initial_delay = defaults[0]
    if poll_interval is None:
        poll_interval = defaults[1]
    if max_wait is None:
        max_wait = defaults[2]
    if stable_checks is None:
        stable_checks = _poll_setting(
            'N2P2_REMOTE_STABLE_CHECKS',
            DEFAULT_REMOTE_STABLE_CHECKS,
        )

    remote_path = _remote_work_path(node)
    if remote_path is None:
        return None

    remote_file = os.path.join(remote_path, filename)
    if logger is not None:
        logger.warning(
            'Waiting for remote %r to stabilise on %r (poll every %ss, '
            '%s stable checks, max wait %ss, initial delay %ss).',
            filename,
            remote_file,
            poll_interval,
            stable_checks,
            max_wait,
            initial_delay,
        )

    deadline = time.monotonic() + max_wait
    if initial_delay > 0:
        time.sleep(initial_delay)

    transport = node.get_transport()
    last_size: int | None = None
    stable_count = 0
    latest_bytes: bytes | None = None

    while time.monotonic() < deadline:
        with transport:
            if not transport.isfile(remote_file):
                last_size = None
                stable_count = 0
            else:
                with tempfile.NamedTemporaryFile(delete=False) as handle:
                    local_path = Path(handle.name)
                try:
                    transport.getfile(remote_file, str(local_path))
                    content = local_path.read_bytes()
                finally:
                    local_path.unlink(missing_ok=True)

                size = len(content)
                if last_size is not None and size == last_size:
                    stable_count += 1
                else:
                    stable_count = 0
                last_size = size
                latest_bytes = content

                if stable_count >= stable_checks:
                    if logger is not None:
                        logger.info(
                            'Remote %r stable at %s bytes after job completion.',
                            filename,
                            size,
                        )
                    return latest_bytes

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_interval, remaining))

    if latest_bytes is not None:
        if logger is not None:
            logger.warning(
                'Returning best-effort remote %r (%s bytes) after timeout.',
                filename,
                len(latest_bytes),
            )
        return latest_bytes

    if logger is not None:
        logger.error(
            'Timed out waiting for stable remote file %r on %r.',
            filename,
            remote_file,
        )
    return None


def fetch_remote_output_file(
    node: CalcJobNode,
    filename: str,
    *,
    initial_delay: int | None = None,
    poll_interval: int | None = None,
    max_wait: int | None = None,
    logger=None,
) -> Path | None:
    """Poll the remote work directory until ``filename`` exists and download it.

    This helps when the scheduler reports the job as finished before output
    files are written on the remote machine (observed with OAR on Dahu).
    """
    defaults = get_remote_poll_settings(node)
    if initial_delay is None:
        initial_delay = defaults[0]
    if poll_interval is None:
        poll_interval = defaults[1]
    if max_wait is None:
        max_wait = defaults[2]

    remote_path = _remote_work_path(node)
    if remote_path is None:
        if logger is not None:
            logger.error('Cannot poll remote output: remote_folder is missing.')
        return None

    remote_file = os.path.join(remote_path, filename)

    if logger is not None:
        logger.warning(
            'Output %r missing locally; polling remote path %r every %ss '
            'for up to %ss (initial delay %ss).',
            filename,
            remote_file,
            poll_interval,
            max_wait,
            initial_delay,
        )

    deadline = time.monotonic() + max_wait
    if initial_delay > 0:
        time.sleep(initial_delay)

    transport = node.get_transport()
    while time.monotonic() < deadline:
        with transport:
            if transport.isfile(remote_file):
                with tempfile.NamedTemporaryFile(delete=False) as handle:
                    local_path = Path(handle.name)
                try:
                    transport.getfile(remote_file, str(local_path))
                except Exception:
                    local_path.unlink(missing_ok=True)
                    raise
                if logger is not None:
                    logger.info('Recovered %r from remote work directory.', filename)
                return local_path

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_interval, remaining))

    if logger is not None:
        logger.error(
            'Timed out waiting for %r on remote path %r after %ss.',
            filename,
            remote_file,
            max_wait,
        )
    return None


def fetch_remote_files_to_directory(
    node: CalcJobNode,
    filenames: Iterable[str],
    destination_dir: os.PathLike[str] | str,
    *,
    initial_delay: int | None = None,
    poll_interval: int | None = None,
    max_wait: int | None = None,
    logger=None,
) -> bool:
    """Poll the remote work directory until all ``filenames`` exist and download them."""
    required = tuple(filenames)
    if not required:
        return True

    defaults = get_remote_poll_settings(node)
    if initial_delay is None:
        initial_delay = defaults[0]
    if poll_interval is None:
        poll_interval = defaults[1]
    if max_wait is None:
        max_wait = defaults[2]

    remote_path = _remote_work_path(node)
    if remote_path is None:
        if logger is not None:
            logger.error('Cannot poll remote outputs: remote_folder is missing.')
        return False

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if logger is not None:
        logger.warning(
            'Weight files %r missing locally; polling remote path %r every %ss '
            'for up to %ss (initial delay %ss).',
            required,
            remote_path,
            poll_interval,
            max_wait,
            initial_delay,
        )

    deadline = time.monotonic() + max_wait
    if initial_delay > 0:
        time.sleep(initial_delay)

    transport = node.get_transport()
    while time.monotonic() < deadline:
        with transport:
            remote_names = set(transport.listdir(remote_path))
            if all(name in remote_names for name in required):
                for name in required:
                    transport.getfile(
                        os.path.join(remote_path, name),
                        str(destination / name),
                    )
                if logger is not None:
                    logger.info(
                        'Recovered weight files %r from remote work directory.',
                        required,
                    )
                return True

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_interval, remaining))

    if logger is not None:
        logger.error(
            'Timed out waiting for weight files %r on remote path %r after %ss.',
            required,
            remote_path,
            max_wait,
        )
    return False
