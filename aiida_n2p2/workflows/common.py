"""Shared helpers for n2p2 workchains."""

from __future__ import annotations

import copy
from typing import Any, Iterable, Type

from aiida.engine import Process


def metadata_dict(metadata_input: Any | None) -> dict:
    """Return a metadata options dict from an optional ``Dict`` input."""
    if metadata_input is None:
        return {}
    return metadata_input.get_dict()


class CalcJobMetadataWorkChainMixin:
    """Accept CalcJob-style ``metadata.options`` on direct WorkChain submits."""

    def _setup_metadata(self, metadata: dict) -> None:
        metadata = copy.copy(metadata)
        metadata.pop('options', None)
        metadata.pop('computer', None)
        super()._setup_metadata(metadata)


def exposed_calcjob_inputs(
    workchain,
    process_class: Type[Process],
    exclude: Iterable[str] = (),
    namespace: str | None = None,
):
    """Return exposed inputs for a nested process, omitting runtime overrides."""
    inputs = workchain.exposed_inputs(process_class, namespace=namespace)
    for key in exclude:
        inputs.pop(key, None)
    return inputs
