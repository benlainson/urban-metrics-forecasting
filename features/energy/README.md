# features/energy/

Owner: Nick.

Feature engineering for hourly PJM-wide demand: UTC calendar fields, demand lags
of 1/24/168 hours, and rolling means over 24/168 hours ending at the previous hour.

Missing timestamps are restored on a complete UTC hourly timeline before shifts.
Invalid demand becomes null rather than removing an hour. Examples with a missing
target or incomplete history are discarded after all features are calculated.
Duplicate timestamps and mixed balancing authorities are rejected.

For demand-only features, run from the repository root:

```bash
.venv/bin/python features/energy/engineer.py --skip-weather
```

This writes `data/processed/energy/features.parquet` and, if benchmarking data
exists, a separate community-area weighting table. That table is not measured
neighborhood hourly demand and is not used by the initial model.

Without `--skip-weather`, the optional weather join accepts `timestamp_utc` or
`period`, normalizes UTC, and requires one selected weather location per hour.
Those observations are joined at the target hour for exploration; do not use
them for forecasting without addressing when they become available.

The model trainer rebuilds demand-only features directly from raw data. See
`models/energy/README.md` for training and verification commands.
