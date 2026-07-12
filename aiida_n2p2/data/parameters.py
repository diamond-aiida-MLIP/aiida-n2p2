"""AiiDA data type for n2p2 ``input.nn`` network configuration."""

from __future__ import annotations

import io
from copy import deepcopy
from pathlib import Path

from aiida.orm import Data, Dict, SinglefileData

from aiida_n2p2.utils.input_nn import (
    dicts_to_lines,
    disable_keyword,
    enable_keyword,
    keyword_states,
    lines_to_dicts,
    parse_input_nn,
    render_input_nn,
    set_keyword,
)


class N2p2Parameters(Data):
    """Structured representation of an ``input.nn`` template."""

    _ATTR_LINES = 'lines'

    @classmethod
    def from_file(cls, filepath: str | Path, *, label: str | None = None) -> N2p2Parameters:
        """Parse an on-disk ``input.nn`` file."""
        path = Path(filepath)
        lines = parse_input_nn(path)
        node = cls()
        if label:
            node.label = label
        node.set_attribute(cls._ATTR_LINES, lines_to_dicts(lines))
        node.set_attribute('source_filename', path.name)
        node.set_attribute('keywords', keyword_states(lines))
        return node

    @classmethod
    def from_lines(cls, lines: list[dict], *, label: str | None = None) -> N2p2Parameters:
        node = cls()
        if label:
            node.label = label
        node.set_attribute(cls._ATTR_LINES, deepcopy(lines))
        node.set_attribute('keywords', keyword_states(dicts_to_lines(lines)))
        return node

    def _get_lines(self):
        return dicts_to_lines(self.get_attribute(self._ATTR_LINES))

    def _store_lines(self, lines) -> None:
        self.set_attribute(self._ATTR_LINES, lines_to_dicts(lines))
        self.set_attribute('keywords', keyword_states(lines))

    def set(self, keyword: str, value: str) -> None:
        """Set an active keyword value."""
        lines = set_keyword(self._get_lines(), keyword, value)
        self._store_lines(lines)

    def enable(self, keyword: str) -> None:
        """Uncomment a flag keyword such as ``use_old_weights_short``."""
        lines = enable_keyword(self._get_lines(), keyword)
        self._store_lines(lines)

    def disable(self, keyword: str) -> None:
        """Comment out a keyword."""
        lines = disable_keyword(self._get_lines(), keyword)
        self._store_lines(lines)

    def apply_overrides(self, overrides: dict | Dict) -> None:
        """Apply a mapping of keyword -> value or ``{'__enable__': [...]}``."""
        if isinstance(overrides, Dict):
            payload = overrides.get_dict()
        else:
            payload = dict(overrides)

        enable_flags = payload.pop('__enable__', [])
        disable_flags = payload.pop('__disable__', [])

        for keyword in enable_flags:
            self.enable(keyword)
        for keyword in disable_flags:
            self.disable(keyword)
        for keyword, value in payload.items():
            self.set(keyword, str(value))

    def render(self) -> str:
        return render_input_nn(self._get_lines())

    def get_singlefile(self) -> SinglefileData:
        return SinglefileData(io.BytesIO(self.render().encode('utf-8')))

    def clone(self) -> N2p2Parameters:
        """Return a mutable copy without storing."""
        return N2p2Parameters.from_lines(self.get_attribute(self._ATTR_LINES))

    def as_dict(self) -> dict:
        return {
            'source_filename': self.get_attribute('source_filename', None),
            'keywords': self.get_attribute('keywords'),
        }

    def prepare_for_restart(
        self,
        *,
        additional_epochs: int | None = None,
    ) -> None:
        """Enable restart keywords and optionally extend ``epochs``."""
        self.enable('use_old_weights_short')
        if additional_epochs is not None:
            keywords = self.get_attribute('keywords')
            current = int(keywords.get('epochs', {}).get('value', '0'))
            self.set('epochs', str(current + additional_epochs))
