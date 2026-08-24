# data/

Storage for all datasets used by the project. Not the source of truth for production data (that's BigQuery / Cloudflare R2 — see root `CLAUDE.md`) — this is the local/working copy used for development and notebooks.

- `raw/` — Unmodified API responses and downloaded files, one subfolder per domain if it gets crowded (e.g. `raw/transit/`, `raw/energy/`, `raw/weather/`). Never edit files here by hand.
- `processed/` — Cleaned, joined, feature-engineered data ready for modeling. Output of scripts in `features/`.
- `schemas/` — Data contracts: column names, types, and expected ranges for each dataset. Update this whenever a raw data source's shape changes so everyone downstream knows.

Large files should not be committed to git — use `.gitignore` for `*.parquet`, `*.csv` etc. and rely on R2/BigQuery for shared storage.
