# aiida-n2p2

AiiDA plugin for the [n2p2](https://github.com/CompPhysVienna/n2p2) neural-network potential package.

**Current version:** 0.4.0

## Features

- **CalcJobs:** `nnp-scaling`, `nnp-train`, `nnp-predict`
- **WorkChains:** scaling, training (manual + automatic restart), LAMMPS validation, full pipeline (`MakeNNPWorkchain`)
- **Data types:** `N2p2Dataset`, `N2p2Parameters` (structured `input.data` / `input.nn`)
- **Training restart (v0.4.0):** automatic relaunch when walltime stops training before target `epochs`; merged learning curves across segments
- **HPC / OAR:** parsers recover outputs from the remote folder when retrieve is premature (Dahu)

## Prerequisites

`pip install -e .` installs the plugin and its Python dependencies (`aiida-core`, `aiida-lammps`, `numpy`) into the environment that is already active. It does not create that environment, an AiiDA profile, or the external executables.

Before the install commands:

- Python 3.10 or newer. With `pip`, activate a virtual environment first (conda or venv). `poetry install` creates its own virtualenv.
- PostgreSQL and RabbitMQ running.
- An AiiDA profile, created with `verdi quicksetup` or `verdi setup`, so that `verdi` already works.
- The n2p2 executables `nnp-scaling`, `nnp-train`, and `nnp-predict` installed on the computer where the calculations run.
- LAMMPS, when the optional validation step is used.

## Installation

```bash
git clone <repo-url> aiida-n2p2
cd aiida-n2p2
pip install -e .          # or: poetry install
verdi daemon restart
```

If entry points are missing after install, remove stale metadata in the repo:

```bash
rm -rf aiida_n2p2.egg-info
pip install -e .
```

Register AiiDA codes (examples for localhost are in `oar-scheduler-main/*_localhost.yml` sibling repo).

## Quick start

```bash
verdi daemon start
cd examples/3.Restart
python restart_train_demo.py          # manual restart: 40 + 40 epochs (localhost)
python auto_restart_dahu.py           # automatic restart on Dahu/OAR
cd ../2.HPC
python wkchain_Al.py                  # full pipeline (default: dahu_parallel)
```

Example layout, env variables, and what is (not) shipped in git:
[`examples/README.md`](examples/README.md).

Restart details: [`examples/3.Restart/README.md`](examples/3.Restart/README.md).

## Tests

```bash
pip install pytest
PYTHONPATH=. pytest -v
```

Details: [`tests/README.md`](tests/README.md).

## Documentation

- Workflow and restart API (PDF): [`docs/aiida-n2p2-workflow.pdf`](docs/aiida-n2p2-workflow.pdf)
- LaTeX sources: [`docs/aiida-n2p2-workflow.tex`](docs/aiida-n2p2-workflow.tex) (standalone),
  [`docs/n2p2_workflow_section.tex`](docs/n2p2_workflow_section.tex) (section for larger documents)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)

### Workflow overview

Vertical layout (fits single-column pages; full TikZ figures in the LaTeX doc):

```
MakeNNPWorkchain
        │
        ▼
  [Prepare inputs]     optional: N2p2Dataset + N2p2Parameters
        │
        ▼
  [Scale]              nnp-scaling  →  scale
        │
        ▼
  [Train]              nnp-train  →  weights, learning curve
        │
        ▼
  [LAMMPS validate]    optional (run_validation=True)
```

## Authors

- **v0.1.0–v0.2.0:** [Akshay Krishna Ammothum Kandy](https://github.com/aksam432) — initial rudimentary plugin (CalcJobs, basic WorkChains, first examples).
- **v0.3.0–v0.4.0:** [Irina Piazza](https://github.com/irina-piazza) — structured inputs (`N2p2Dataset`, `N2p2Parameters`), training restart, HPC/OAR integration, tests, and documentation.

## License

MIT
