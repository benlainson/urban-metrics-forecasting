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
