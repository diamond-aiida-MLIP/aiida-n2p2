# Regression tests (Al, dump-949)

Fast regression tests for the aiida-n2p2 plugin. They reuse pre-computed
outputs from the successful Al run `MakeNNPWorkchain<949>` and do **not**
re-run n2p2 or LAMMPS.

## Plugin layout

```
aiida_n2p2/
├── calculations/   nnpScaling, nnpTraining, nnpPredict
├── parsers/        matching parsers
└── workflows/
    ├── scale.py            WorkflowFactory('n2p2.scale')
    ├── train.py            WorkflowFactory('n2p2.train')
    ├── validate_lammps.py  WorkflowFactory('n2p2.validate_lammps')
    └── make_potential.py   WorkflowFactory('n2p2.make_potential')
```

Each step can be submitted independently, or composed through
`MakeNNPWorkchain`.

## Memory-conscious defaults

- **Training:** only `learning-curve.out` is stored permanently; all
  `weights.*.out` files are retrieved temporarily and discarded after the
  parser selects the best epoch.
- **LAMMPS validation:** trajectory files (`*.lammpstrj`) are **not**
  retrieved unless `validation.retrieve_trajectory = Bool(True)`.

## Run

```bash
pip install pytest
PYTHONPATH=. pytest -v
PYTHONPATH=. pytest -v --regression-report
python tests/show_reference.py
```

## Interpreting results

| Result | Meaning |
|--------|---------|
| **PASSED** | Parsed values still match the Al reference |
| **FAILED** | Behaviour changed — fix a bug or update `REFERENCE.json` deliberately |
