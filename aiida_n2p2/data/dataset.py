"""AiiDA data type wrapping an n2p2 ``input.data`` training set."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from aiida.orm import Data, SinglefileData

BEGIN_PATTERN = re.compile(r'^\s*begin\s*$', re.MULTILINE)
ELEMENT_PATTERN = re.compile(r'^\s*atom\s+', re.MULTILINE)


class N2p2Dataset(Data):
    """Training-set container with lightweight metadata."""

    _FILENAME = 'input.data'

    @classmethod
    def from_file(cls, filepath: str | Path, *, label: str | None = None) -> N2p2Dataset:
        """Build a dataset node from an on-disk ``input.data`` file."""
        path = Path(filepath)
        content = path.read_bytes()
        node = cls()
        if label:
            node.label = label
        node.set_attribute('md5', hashlib.md5(content).hexdigest())
        node.set_attribute('n_structures', cls._count_structures(content.decode('utf-8', errors='replace')))
        node.set_attribute('source_filename', path.name)
        node.put_object_from_file(str(path), cls._FILENAME)
        return node

    @staticmethod
    def _count_structures(text: str) -> int:
        return len(BEGIN_PATTERN.findall(text))

    def get_singlefile(self) -> SinglefileData:
        """Return the dataset as ``SinglefileData`` for CalcJobs."""
        with self.base.repository.open(self._FILENAME, 'rb') as handle:
            return SinglefileData(file=handle, filename=self._FILENAME)

    def as_dict(self) -> dict:
        return {
            'md5': self.get_attribute('md5'),
            'n_structures': self.get_attribute('n_structures'),
            'source_filename': self.get_attribute('source_filename'),
        }
