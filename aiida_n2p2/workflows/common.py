"""Shared helpers for n2p2 workchains."""

from __future__ import annotations

from typing import Any


def metadata_dict(metadata_input: Any | None) -> dict:
    """Return a metadata options dict from an optional ``Dict`` input."""
    if metadata_input is None:
        return {}
    return metadata_input.get_dict()
