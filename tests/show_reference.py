#!/usr/bin/env python3
"""Print the golden regression reference (no pytest required).

Usage:
    python tests/show_reference.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FIXTURES_AL = Path(__file__).parent / 'fixtures' / 'al'
REFERENCE_PATH = FIXTURES_AL / 'REFERENCE.json'


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    with REFERENCE_PATH.open(encoding='utf-8') as handle:
        ref = json.load(handle)

    print('=' * 60)
    print('aiida-n2p2 regression reference')
    print('=' * 60)
    print(ref['description'])
    print()
    print(f"Element .............. {ref['element']} (Z={ref['atomic_number']})")
    print(f"Source workchain PK ... {ref['source_workchain_pk']}")
    print(f"Source date .......... {ref['source_date']}")
    print()
    print('Scaling step')
    print(f"  file ............... {ref['scaling']['file']}")
    print(f"  md5 ................ {ref['scaling']['md5']}")
    print(f"  on-disk md5 ........ {md5(FIXTURES_AL / ref['scaling']['file'])}")
    print()
    print('Training step')
    print(f"  best epoch ......... {ref['training']['best_epoch']}")
    print(f"  best test RMSE ..... {ref['training']['best_test_rmse']}")
    print(f"  best weights ....... {ref['training']['best_weights_file']}")
    print(f"  weights md5 ........ {ref['training']['best_weights_md5']}")
    print(
        f"  on-disk md5 ........ "
        f"{md5(FIXTURES_AL / ref['training']['best_weights_file'])}"
    )
    print()
    print('Run tests:  pytest -v')
    print('With report: pytest -v --regression-report')
    print('=' * 60)


if __name__ == '__main__':
    main()
