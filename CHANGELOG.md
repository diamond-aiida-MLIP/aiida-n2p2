## v0.2.0 (2025-04-18)

### Feat

- Metadata access from workflows
- Dynamic naming of weight files based on atomic number
- Best weights chosen for lowest test set error and parsing of atomic numbers

### Fix

- The weights named in correct format for lammps

### Refactor

- Single dict  input for n2p2 workchain

## v0.1.0 (2025-02-06)

### Feat

- Dynamic naming of weight files based on atomic number
- Best weights chosen for lowest test set error and parsing of atomic numbers
- Example to train a MLP using aiida-n2p2 workflow
- Workflow for training MLP potential and performing MD with LAMMPS
- Calcjob interface  for nnp-predict
- Basic interface for nnp-train function
- Calculation interface for nnp-scaling operation

### Perf

- Training default changed to 8 cores and specific epoch for Aluminium liquid set
- Default number of cores changed to 8
