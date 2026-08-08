# Examples

Runnable scripts and reference inputs for **aiida-n2p2** v0.4.0.

Network settings (`epochs`, hidden layers, nodes, activation) can be changed **without
editing `input.nn`** using `N2p2Parameters` and optional environment variables (see
[Common environment variables](#common-environment-variables)).

## Directory layout

| Folder | Purpose | Main script(s) |
|--------|---------|----------------|
| [`1.Al/`](1.Al/) | Jupyter tutorial (liquid Al) | `AiidA-n2p2_demo.ipynb` |
| [`1.Boron/`](1.Boron/) | Full pipeline on Boron (HPC) | `aiida-n2p2_demo.py` |
| [`2.HPC/`](2.HPC/) | Full pipeline on Al (scale → train → LAMMPS) | `wkchain_Al.py` |
| [`3.Restart/`](3.Restart/) | Manual and automatic training restart | `restart_train_demo.py`, `auto_restart_*.py` |

### What is versioned

Each example folder keeps **scripts**, **`input.nn`**, **`in.lmp`**, **`222_IN.data`**
(LAMMPS structure), and (where small enough) **`input.data`**:

| File | `1.Al` | `2.HPC` | `3.Restart` | `1.Boron` |
|------|--------|---------|-------------|-----------|
| `input.nn` | yes | yes | yes (pytest fixture) | yes |
| `input.data` | yes (~18 MB) | yes (~18 MB) | yes (minimal) | **no** (~939 MB, local only) |
| `in.lmp`, `222_IN.data` | yes | yes | — | yes |
| Python demos | notebook | `wkchain_Al.py` | restart scripts | `aiida-n2p2_demo.py` |

### What is **not** versioned (see root `.gitignore`)

- AiiDA dumps: `dump-*/`
- n2p2 outputs: `*.out`, `weights.*`, `scaling.data`, `learning-curve*.out`, …
- Generated plots: `examples/**/*.png`
- Boron local research: `analysis/`, `optimization/`, large `input.data`
- Scheduler / AiiDA staging files inside dumps

If you ran calculations locally, those artifacts remain on disk but are ignored by git.

---

## Quick start

```bash
# From repo root
pip install -e .
verdi daemon start

# Manual restart (localhost, 40 + 40 epochs)
cd examples/3.Restart && python restart_train_demo.py

# Full Al pipeline (default: dahu_parallel)
cd examples/2.HPC && python wkchain_Al.py

# Boron pipeline (requires local input.data — see below)
cd examples/1.Boron && python aiida-n2p2_demo.py
```

Details for restart modes: [`3.Restart/README.md`](3.Restart/README.md).

---

## Common environment variables

Used by `wkchain_Al.py`, `aiida-n2p2_demo.py`, and restart scripts:

| Variable | Effect |
|----------|--------|
| `N2P2_INPUT_DIR` | Directory with `input.data` / `input.nn` (default: script folder or `2.HPC`) |
| `N2P2_COMPUTER` | AiiDA computer label (default: `dahu_parallel`) |
| `N2P2_SCALE_CODE` | e.g. `n2p2_scale@localhost` |
| `N2P2_TRAIN_CODE` | e.g. `n2p2_train@localhost` |
| `N2P2_LAMMPS_CODE` | e.g. `lammps@localhost` |
| `N2P2_OAR_PROJECT` | OAR account on Dahu (default: `pr-diamond`) |
| `N2P2_TARGET_EPOCHS` | Override `epochs` in `input.nn` |
| `N2P2_HIDDEN_LAYERS` | → `global_hidden_layers_short` |
| `N2P2_NODES` | → `global_nodes_short` (e.g. `'5 5'`) |
| `N2P2_ACTIVATION` | → `global_activation_short` (e.g. `'p p l'`) |
| `N2P2_SCALE_NBIN` | Scaling bins (default: 100) |
| `N2P2_TRAIN_MPI` | MPI processes for training |
| `N2P2_WALLTIME_SECONDS` | Job walltime |
| `N2P2_RUN_VALIDATION` | `true` / `false` — run LAMMPS step in full pipeline |

Restart-specific variables are documented in [`3.Restart/README.md`](3.Restart/README.md).

---

## Boron: local `input.data`

The Boron training set (`input.data`, ~939 MB) is **not** in the repository.
Place your dataset as:

```text
examples/1.Boron/input.data
```

The demo uses the same `N2p2Parameters` workflow as `wkchain_Al.py`; only
`input.nn`, `in.lmp`, and `222_IN.data` are shipped with the plugin.

---

## Programmatic `input.nn` (no manual file edit)

```python
from aiida.plugins import DataFactory

N2p2Parameters = DataFactory('n2p2.parameters')
params = N2p2Parameters.from_file('input.nn')
params.set('epochs', '300')
params.set('global_nodes_short', '5 5')
params.store()
input_nn = params.get_singlefile()
```

Full pipeline scripts pass `prepare={'parameters': params, 'overrides': ...}` to
`MakeNNPWorkchain` — see `2.HPC/wkchain_Al.py` and `1.Boron/aiida-n2p2_demo.py`.
