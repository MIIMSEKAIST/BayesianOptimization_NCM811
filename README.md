# Bayesian Optimization for NCM811 Synthesis

An iterative, experiment-in-the-loop Bayesian optimization workflow for identifying NCM811 synthesis conditions that maximize initial discharge capacity.

The repository combines a random-forest surrogate model, expected-improvement acquisition, categorical precursor selection, bounded numerical process variables, strict iteration guards, and per-iteration diagnostic visualizations.

> [!IMPORTANT]
> The current public repository is not yet reproducible from a clean clone. The source imports `config.py`, but that file and the initial training CSV referenced by it are not included. See [Required local files](#required-local-files) before running the workflow.

## Overview

The optimizer searches across four categorical precursor choices and six numerical synthesis variables. Each Bayesian optimization iteration proposes three previously untested experiments. After those experiments are performed, their measured initial discharge capacities are entered into the working CSV and returned to the optimizer.

```mermaid
flowchart LR
    A[Initial dataset] --> B[Train optimizer]
    B --> C[Suggest 3 experiments]
    C --> D[Run experiments]
    D --> E[Enter measured capacity]
    E --> F[Update optimizer]
    F --> G[Generate diagnostics]
    G --> C
```

### Optimization variables

| Type | Variables |
| --- | --- |
| Categorical | `Li_Precursor`, `Ni_Precursor`, `Co_Precursor`, `Mn_Precursor` |
| Numerical | `Hydrothermal_Temperature`, `Hydrothermal_Time`, `First_Calcination_Temperature`, `First_Calcination_Time`, `Second_Calcination_Temperature`, `Second_Calcination_Time` |
| Objective | `Discharge_Capacity` in mAh g⁻¹, as configured by `TARGET_VARIABLE` |

Categorical variables are ordinal-encoded for model input. The numerical search bounds and exact objective transform are defined in the local `config.py`. The optimizer minimizes the negative transformed capacity, which is equivalent to seeking higher discharge capacity when the configured transform is monotonic.

## Method

- **Surrogate model:** random forest (`skopt.learning.RandomForestRegressor`)
- **Acquisition function:** expected improvement (`EI`)
- **Batch size:** three candidate experiments per iteration
- **Duplicate control:** candidates already present in the encoded dataset are rejected
- **Candidate ranking:** a separate 500-tree `sklearn.ensemble.RandomForestRegressor` ranks pending candidates by predicted capacity
- **Validation:** K-fold cross-validation and out-of-bag diagnostics
- **Reproducibility:** optimizer, encoder, iteration counter, accumulated data, and per-iteration plots are persisted

## Repository structure

| Path | Purpose |
| --- | --- |
| `NCM_811_RF.py` | Command-line entry point |
| `models.py` | Implements `train`, `iterate`, `update`, `best`, `visualize`, and `loop` modes |
| `optimizer_utils.py` | Defines the mixed search space, initializes the optimizer, and generates unique candidates |
| `protocol.py` | Enforces train-once and sequential iterate/update rules |
| `utils.py` | CSV, pickle, encoding, iteration, and snapshot helpers |
| `viz.py` | Convergence, feature-importance, heatmap, parity, residual, and reliability plots |
| `guidance.txt` | Short command sequence for repeated experimental iterations |
| `iteration.txt` | Current iteration counter; this is runtime state, not a standalone result |
| `rf_optimized_model.pkl` | Serialized optimizer artifact included in the repository |
| `rf_encoder.pkl` | Serialized categorical encoder included in the repository |

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn
- scikit-optimize

Install the Python dependencies with:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn scikit-optimize
```

For a reproducible release, add a pinned `requirements.txt` or environment file because serialized scikit-learn and scikit-optimize objects may not load correctly across library versions.

## Required local files

### 1. `config.py`

All modules import project settings from `config.py`. The file must define at least the following names:

```text
LOGGER
TRAIN_DATA_FILE
DATA_SAVE_FILE
OPTIMIZER_FILE
ENCODER_FILE
ITERATION_FILE
TARGET_VARIABLE
PLOTS_DIR
SEED
BATCH_SIZE
CAT_COLS
NUM_COLS
ALL_OPT_COLS
NUMERICAL_BOUNDS
log_t
```

`NUMERICAL_BOUNDS` must map each numerical variable to a `(minimum, maximum)` pair. `CAT_COLS`, `NUM_COLS`, and `ALL_OPT_COLS` must follow the column names and order shown above. The function `log_t` must reproduce the objective transform used to create the stored optimizer.

### 2. Initial training CSV

The file referenced by `TRAIN_DATA_FILE` must contain one row per completed experiment and include the four precursor columns, six numerical synthesis columns, and the target-capacity column. A `doi` column is optional and is removed during loading.

Minimal schema:

```csv
Li_Precursor,Ni_Precursor,Co_Precursor,Mn_Precursor,Hydrothermal_Temperature,Hydrothermal_Time,First_Calcination_Temperature,First_Calcination_Time,Second_Calcination_Temperature,Second_Calcination_Time,Discharge_Capacity
```

Rows outside the configured numerical bounds are excluded during loading. Missing precursor categories, numerical inputs, or initial target values are not supported during initial training.

## Usage

Run all commands from the repository root.

### 1. Initialize the optimizer

Run this once for a new optimization campaign:

```bash
python NCM_811_RF.py --mode train --verbose
```

Training fits the categorical encoder, initializes the random-forest Bayesian optimizer with the initial dataset, writes the optimizer and encoder pickle files, creates the accumulated-data CSV, and sets the iteration counter to 1.

The protocol intentionally blocks a second training run when saved state already exists. Back up the campaign before deleting any saved state to restart.

### 2. Generate the next experimental batch

```bash
python NCM_811_RF.py --mode iterate
```

This appends three unique candidate rows to the accumulated-data CSV. Their target-capacity cells are left empty for manual entry after the experiments are completed.

### 3. Inspect the predicted ranking

```bash
python NCM_811_RF.py --mode best --top 3
```

This fits a random-forest regressor to all completed experiments and ranks pending candidates from the latest iteration by predicted discharge capacity. The ranking is printed to the log; it does not replace experimental validation.

### 4. Perform the experiments and enter results

Open the CSV referenced by `DATA_SAVE_FILE`, find the three rows for the current iteration, and enter the measured values in `Discharge_Capacity`. Do not change encoded state files or the `Iteration` values.

### 5. Update the optimizer

```bash
python NCM_811_RF.py --mode update
```

The protocol permits an update only when every candidate in the current iteration has a measured target. A successful update saves the optimizer and advances `iteration.txt`.

### 6. Generate figures and diagnostics

```bash
python NCM_811_RF.py --mode visualize
```

Outputs are written beneath:

```text
<PLOTS_DIR>/iter_000/
<PLOTS_DIR>/iter_001/
<PLOTS_DIR>/iter_002/
...
```

Depending on the amount of completed data, each snapshot may include:

- best-capacity convergence
- surrogate learning curve
- training-versus-suggested capacity distribution
- random-forest feature importance
- Li/Ni and all pairwise precursor heatmaps
- cross-validated parity and residual plots
- uncertainty-versus-error reliability plot
- out-of-bag parity, residual, and interval-calibration plots
- `MANIFEST.txt` with the snapshot size and best measured capacity

Some diagnostics require at least 10 completed experiments; the reliability suite requires at least 3.

### Optional interactive loop

```bash
python NCM_811_RF.py --mode loop --iterations 5
```

The loop generates a batch, pauses while experiments are run and results are entered, and then updates the optimizer. It does not automatically run `best` or `visualize`.

## Protocol safeguards

The code prevents common state-management errors:

- training more than once without resetting saved state
- opening a new iteration while earlier measurements are missing
- generating a second candidate batch for an already-open iteration
- updating before all current candidates have measured capacities
- updating before candidate rows have been generated

Treat the accumulated CSV, optimizer pickle, encoder pickle, and `iteration.txt` as one synchronized campaign state. Back them up together before manual editing or migration.

## Current limitations

- `config.py`, the initial training CSV, and the accumulated experimental CSV are absent from the public repository, so the current checkout cannot run as-is.
- `requirements.txt`, automated tests, and a sample configuration are not yet provided.
- The numerical search bounds and categorical choices cannot be inferred from the committed files alone.
- `update` currently passes all completed rows to the optimizer on each call, not only newly completed rows. Verify whether repeated observations are intended before using the workflow for a new campaign.
- Pickle files should only be loaded from trusted sources. Python pickle can execute arbitrary code during deserialization.
- The RF tree-to-tree spread used in the reliability plots is a relative uncertainty proxy, not a calibrated Bayesian posterior uncertainty.
- With small experimental datasets, cross-validation, feature importance, and candidate ranking can be unstable and should be interpreted cautiously.

## Reproducing or continuing the included campaign

The included `rf_optimized_model.pkl`, `rf_encoder.pkl`, and `iteration.txt` represent only part of the campaign state. To continue it safely, use the matching `config.py`, accumulated CSV, dependency versions, and exact objective transform from the original environment. Do not combine the included pickle files with a newly fitted encoder or differently ordered feature columns.

## Citation

The associated publication and citation information will be added after publication. If you use this workflow before then, please cite this repository and record the commit SHA used in your analysis.

## License

No license file is currently included. Unless a license is added, reuse and redistribution permissions are not granted by default. Contact the repository owner for permission.

## Contact

Maintained by [MII Lab, KAIST](https://github.com/MIIMSEKAIST).
