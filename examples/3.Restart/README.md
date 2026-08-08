# Training restart examples (v0.4.0)

Minimal **scale + train** examples without LAMMPS validation.

**Input files:** all runnable scripts load the **Aluminium** dataset from
`examples/2.HPC/input.data` and `examples/2.HPC/input.nn` (override with
`N2P2_INPUT_DIR`). The small `input.data` / `input.nn` in this folder are
**pytest fixtures only** — real n2p2 runs with them produce zero scaling and NaN RMSE.

## Two restart modes (same CalcJob engine, different orchestration)

| | **Manual restart** | **Automatic restart** |
|---|---|---|
| **API** | `is_restart=True` + `previous_session=wc` | `auto_restart=True` |
| **Who decides** | You (second `submit` / script step) | WorkChain (internal loop) |
| **AiiDA submits** | **2** `N2p2TrainWorkChain` nodes | **1** WorkChain, N CalcJobs |
| **Target epochs** | Split via `N2P2_TARGET_EPOCHS` (default 80 → 40+40) | Fixed in `parameters` / `input.nn` |
| **Example script** | `restart_train_demo.py` | `auto_restart_dahu.py`, `auto_restart_train_demo.py` |
| **Best environment** | Laptop or HPC | **HPC with scheduler** (walltime enforced) |

Both modes enable `use_old_weights_short`, copy `last_weights` to `weights.ZZZ.data`, and merge learning curves.

### `epochs` in restart segments

- **Segment 1:** `input.nn` gets `epochs = N2P2_EPOCHS_FIRST` (default 40).
- **Segment 2 (manual):** WorkChain sets `epochs = N2P2_EPOCHS_RESTART` (**remaining**, default 40), not the cumulative total 80.
- **Auto-restart:** each new segment gets `epochs = remaining` toward the global target.

---

## Example 1 — Manual restart (`restart_train_demo.py`)

Default **40 + 40 = 80** total epochs. Submits **two** training WorkChains.

```bash
cd examples/3.Restart
verdi daemon start
python restart_train_demo.py
```

Environment variables:

```bash
export N2P2_TARGET_EPOCHS=80
export N2P2_EPOCHS_FIRST=40      # optional override
export N2P2_EPOCHS_RESTART=40    # optional override
export N2P2_SCALE_CODE='n2p2_scale@localhost'
export N2P2_TRAIN_CODE='n2p2_train@localhost'
```

Resume **only segment 2** after a successful run 1:

```bash
export N2P2_RESUME_RUN1_PK=<pk_of_first_TrainWorkChain>
python restart_train_demo.py
```

Success: **two** `N2p2TrainWorkChain` nodes; second outputs `learning_curve_plot_data['n_runs'] == 2`.

---

## Example 2 — Automatic restart on Dahu (`auto_restart_dahu.py`)

Preconfigured for `n2p2_scale@dahu_parallel`, `n2p2_train@dahu_parallel`, OAR project `pr-diamond`.

```bash
verdi daemon start
python auto_restart_dahu.py
```

Defaults: `N2P2_TARGET_EPOCHS=300`, `N2P2_WALLTIME_SECONDS=360`, MPI 4 procs.

Optional tuning:

```bash
export N2P2_WALLTIME_SECONDS=360
export N2P2_TARGET_EPOCHS=300
export N2P2_OAR_PROJECT=pr-diamond
python auto_restart_dahu.py
```

Generic scheduler version: `auto_restart_train_demo.py` (localhost or any computer).

---

## Example 3 — Full pipeline (`examples/2.HPC/wkchain_Al.py`)

Scale → train → LAMMPS validation. Uses `N2p2Parameters` (default **200 epochs**
from `input.nn`; override without editing the file).

```bash
cd examples/2.HPC
python wkchain_Al.py                    # default: dahu_parallel
export N2P2_COMPUTER=localhost          # local smoke test
export N2P2_TARGET_EPOCHS=300           # optional
export N2P2_NODES='5 5'                 # optional NN architecture
python wkchain_Al.py
```

Boron variant: `examples/1.Boron/aiida-n2p2_demo.py` (requires local
`input.data` — see [`examples/README.md`](../README.md)).

---

## Prerequisites

- Plugin installed: `pip install -e .` from repo root; `verdi daemon restart`
- AiiDA codes, e.g. `n2p2_scale@localhost`, `n2p2_train@localhost`, `lammps@localhost`
- For Dahu: VPN/SSH via Gricad jump host (`ssh dahu.ciment` must work)

---

## Tests (no real n2p2 run)

```bash
PYTHONPATH=. pytest tests/test_restart.py tests/test_training_state.py -v
```
