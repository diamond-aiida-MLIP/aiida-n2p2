## v0.4.0 (2026-08-08)

### Feat

- Automatic training restart in `N2p2TrainWorkChain` (`auto_restart`, `max_auto_restarts`)
- Remote output recovery for scaling and training parsers (OAR premature retrieve)
- `training_summary` with `target_epochs`, `training_completed`, weight/curve epoch fields
- Examples: `examples/3.Restart/` (manual + auto restart), `wkchain_Al.py` and
  `aiida-n2p2_demo.py` with `N2p2Parameters` and env-based NN/epochs overrides
- Regression tests for restart wiring and training state helpers

### Docs

- `examples/README.md`: layout, versioned files, env variables, Boron dataset note
- Tightened `.gitignore` (AiiDA dumps, n2p2 outputs, Boron local research artifacts)

### Fix

- Manual restart: `additional_epochs` sets **remaining** epochs in restart `input.nn` (not total target)
- `previous_session` input uses `non_db=True` (ProcessNode link, not serialized as Data)
- WorkChain `finalize_outputs` stores output nodes before exposure
- Scale / train WorkChain runtime input wiring (`exposed_calcjob_inputs`)

## v0.3.0

### Feat

- `N2p2Dataset` and `N2p2Parameters` structured data types
- Manual training restart (`is_restart`, `previous_session`, `additional_epochs`)
- Merged learning curves and `learning_curve_plot_data` across restart segments
- `N2p2PrepareInputsWorkChain`

## v0.2.0 (2025-04-18)

### Feat

- Metadata access from workflows
- Dynamic naming of weight files based on atomic number
- Best weights chosen for lowest test set error and parsing of atomic numbers

### Fix

- The weights named in correct format for lammps

### Refactor

- Single dict input for n2p2 workchain

## v0.1.0 (2025-02-06)

### Feat

- Dynamic naming of weight files based on atomic number
- Best weights chosen for lowest test set error and parsing of atomic numbers
- Example to train a MLP using aiida-n2p2 workflow
- Workflow for training MLP potential and performing MD with LAMMPS
- Calcjob interface for nnp-predict
- Basic interface for nnp-train function
- Calculation interface for nnp-scaling operation

### Perf

- Training default changed to 8 cores and specific epoch for Aluminium liquid set
- Default number of cores changed to 8
