# ingestion/energy/

Owner: Nick.

Scripts that pull EIA / utility API data (grid load by neighborhood, energy benchmarking) and write to `data/raw/`.

## EIA demand setup

Install dependencies in the project environment:

```bash
.venv/bin/python -m pip install -r ingestion/energy/requirements.txt
```

Put `EIA_API_KEY=your_key` in `.env` at the repository root (next to
`CLAUDE.md`). The EIA ingestion script loads that file automatically; an existing
shell or CI environment variable takes precedence. `.env` is gitignored.

Run from the repository root:

```bash
.venv/bin/python ingestion/energy/fetch_eia_demand.py --start 2020-01-01 --end 2026-01-01
```

This replaces `data/raw/energy/eia_demand_pjm.parquet` with the requested range.

## Commonwealth Edison demand

For a Chicago-focused target, fetch hourly demand for EIA's Commonwealth Edison
subregion (`CE`) inside PJM:

```bash
.venv/bin/python ingestion/energy/fetch_eia_subregion_demand.py \
  --parent PJM --subregion CE --start 2020-01-01 --end 2026-01-01
```

This writes `data/raw/energy/eia_demand_comed.parquet`. It is ComEd-zone demand,
not neighborhood-level demand. Train it with:

```bash
.venv/bin/python -m models.energy.train_xgboost \
  --input data/raw/energy/eia_demand_comed.parquet \
  --output data/processed/energy/xgboost_comed \
  --validation-start 2024-01-01 --test-start 2025-01-01 --test-end 2026-01-01 \
  --max-demand 50000 --n-estimators 2400
```

The 50,000 MWh screening limit is deliberately above observed plausible ComEd
peaks while removing a six-figure source error found in the current pull. The
training report records the limit and excluded-hour coverage.
The larger tree cap lets validation-based early stopping choose the ComEd run's
tree count instead of stopping only because it reached the default 1,200-tree cap.
