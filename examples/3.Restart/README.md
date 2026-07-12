# Training restart demo (v0.3.0)

Minimal **scale + train + restart** example for interactive smoke testing.

## What it does

1. Loads a tiny Al training set (`input.data`, 2 structures) and `input.nn` (20 epochs).
2. Runs `N2p2ScaleWorkChain` once.
3. Runs `N2p2TrainWorkChain` (run 1, 20 epochs).
4. Restarts with `is_restart=True`, `previous_session=run1`, and
   `additional_epochs=20` via `N2p2Parameters.prepare_for_restart()`.
5. Checks merged learning-curve metadata and writes `learning_curve_merged.png`.

No LAMMPS validation is involved.

## Prerequisites

- AiiDA profile with n2p2 codes (`nnp-scaling`, `nnp-train`).
- Default code labels: `n2p2@local`. Override if needed:

```bash
export N2P2_SCALE_CODE='n2p2_scale@dahu_parallel'
export N2P2_TRAIN_CODE='n2p2_train@dahu_parallel'
```

## Run

```bash
cd examples/3.Restart
python restart_train_demo.py
```

Expected runtime: seconds to a few minutes on a laptop (2 structures, 40 epochs total).

## Notes

- This demonstrates **deliberate multi-session restart** (both runs finish successfully).
- Restart after scheduler walltime limits is planned for a future release; see the
  plugin documentation for details.
