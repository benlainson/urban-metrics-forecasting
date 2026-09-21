# models/energy/

Owner: Nick.

One-hour-ahead EIA electricity demand forecasting. The trainer supports one
balancing-authority series such as PJM or one subregion such as Commonwealth
Edison (`parent=PJM`, `subba=CE`). Neighborhood weights are a separate
approximation and are not training targets or model inputs.

## Run training

From the repository root (`urban-metrics-forecasting`), using Python 3.11+:

```bash
# Use the existing environment, or create it with: python3 -m venv .venv
.venv/bin/python -m pip install -r models/energy/requirements.txt
.venv/bin/python -m models.energy.train_xgboost
```

On macOS, XGBoost also needs OpenMP: `brew install libomp`. If importing XGBoost
reports that `libomp.dylib` is missing, install that runtime and rerun.

The raw input must already exist at `data/raw/energy/eia_demand_pjm.parquet`.
The trainer rebuilds features from raw data every time, so you do not need to run
the feature script first. Training is local, CPU-only, and uses no API key or
external tracking service.

For the Chicago-area ComEd-zone experiment, use
`data/raw/energy/eia_demand_comed.parquet`, a separate output directory, and
`--max-demand 50000`. The ingestion README contains the full command.

Outputs are saved under `data/processed/energy/xgboost/` (gitignored). Re-running
overwrites that run; use `--output data/processed/energy/another_run` to keep it.

## Check the result

```bash
# Verify time gaps, missing values, no current/future target leakage, and splits:
.venv/bin/python -m unittest discover -s test/energy -v

# Reload the saved model and reproduce all held-out predictions and error scores:
.venv/bin/python -m models.energy.check_model
```

The second command should print two `PASS` lines. For a different run, pass the
same `--output` directory to both the trainer and checker.

Open these files in the run directory:

| File | What to check |
|---|---|
| `report.md` | Split dates, usable-hour coverage, model-versus-baseline scores, and limits |
| `test_forecast.png` | Predicted versus actual demand and errors by Chicago hour |
| `test_predictions.csv` | Each target hour, forecast origin, actual, and four predictions |
| `metrics.json` | Full scores, parameters, ordered features, versions, and raw-input checksum |
| `feature_importance.csv` | Which features the fitted trees use most |
| `model.json` | Portable saved XGBoost model |
| `test_features.parquet` | Exact inputs used to reproduce held-out predictions |

**MAE** is the average absolute error. **RMSE** gives larger errors more weight.
Both are in the raw series' units, MWh per hourly interval; lower is better.
The report compares XGBoost against demand from the previous hour, the same UTC
hour yesterday, and the same UTC hour last week. Check the strongest baseline,
not just the weakest one. Peak-hour MAE uses a threshold determined from the
training data's 90th percentile, not from the test set.

## Forecast and evaluation contract

- A row with `period=t` targets `value[t]`. The latest demand input is `value[t-1]`.
  This assumes prior-hour demand is available at each forecast origin; actual
  publication delays must be accounted for before production use.
- Features: UTC calendar fields, lags of 1/24/168 hours, and backward-looking
  24/168-hour means ending at t-1. `value` is never an input feature.
- Raw naive timestamps are interpreted as UTC, matching the EIA schema.
  Missing timestamps are inserted before any shifts. Missing, negative,
  nonfinite, and >=300,000 readings become null. No interpolation occurs.
- Examples with missing targets or incomplete history are excluded only after
  feature calculation. Metrics cover the remaining hours; the report exposes
  coverage rather than implying every scheduled hour was scored.
- Train: `[2020-01-01, 2023-01-01)`; validation: `[2023-01-01, 2023-12-01)`;
  test: `[2023-12-01, 2024-01-01)`. Boundaries are UTC and end-exclusive.
- Validation RMSE selects the boosting round with early stopping. The test set
  is never passed to `fit`, used for tuning, or included in a refit. The saved
  estimator's predictions use its best iteration, per the
  [XGBoost prediction documentation](https://xgboost.readthedocs.io/en/stable/prediction.html).
- This is a rolling one-step evaluation: actual prior hours continue to become
  available during validation/test. It is not a forecast of the entire month
  issued at the start of December.
- Weather is deliberately excluded for this first baseline. Join timestamps and
  forecast-time availability must be settled before adding weather inputs.
- December alone does not demonstrate year-round or current-day accuracy.
  Once you use this test result to choose changes, reserve a fresh holdout.

Split boundaries and inputs can be configured with `--train-start`,
`--validation-start`, `--test-start`, `--test-end`, and `--input`. All partitions
must contain usable examples. `--n-estimators` sets the maximum boosting rounds
(default 1,200); `--max-demand` controls the cleaning threshold.

## Load the model in Python

```python
import json
from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor

run = Path("data/processed/energy/xgboost")
columns = json.loads((run / "metrics.json").read_text())["feature_columns"]
model = XGBRegressor()
model.load_model(run / "model.json")
inputs = pd.read_parquet(run / "test_features.parquet")
predictions = model.predict(inputs[columns])
```

For a future live prediction, construct the same ordered features using only
past demand and the target hour's calendar. The training feature builder also
requires targets for evaluation, so it is not yet a live serving interface.
