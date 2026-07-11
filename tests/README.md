# Regression tests (Al, dump-949)

These tests check that **parsers and plugin wiring keep the same behaviour**
as the successful HPC run `MakeNNPWorkchain<949>` (Aluminum).

They are **fast** (seconds): they reuse pre-computed output files, they do **not**
re-run n2p2 or LAMMPS.

## Layout

```
tests/
├── fixtures/al/          # Small golden files copied from dump-949
│   ├── REFERENCE.json    # Expected md5, epoch, RMSE (machine-readable)
│   ├── scaling.data
│   ├── learning-curve.out
│   └── weights.013.000182.out
├── show_reference.py     # Human-readable summary (no pytest)
└── test_*.py
```

The full dump stays in `examples/2.HPC/dump-MakeNNPWorkchain-949/` as
documentation/archive. Only a few kilobytes are copied here for CI.

## Run

```bash
pip install -e ".[testing]"   # or: pip install pytest numpy aiida-core
pytest -v
pytest -v --regression-report  # extra summary at the end
python tests/show_reference.py # print expected values anytime
pytest -v -s tests/test_reference_manifest.py  # print REFERENCE.json
```

## Interpreting results

| Result | Meaning |
|--------|---------|
| **PASSED** | Parsed values still match the Al reference |
| **FAILED** | Behaviour changed — fix a bug **or** update `REFERENCE.json` on purpose |

If you refactor internal code and tests still pass, the external behaviour is unchanged.
