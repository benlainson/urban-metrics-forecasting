# Urban Pulse — Project Context

Shared context for Claude Code (and for us) across this repo. Keep this updated as decisions get made — it's the source of truth for "why" behind the code, not a duplicate of `URBAN_PULSE_PROJECT_PLAN.md` (the 60-day build plan) or per-folder `README.md` files (what each folder is for).

## Team & Ownership

Three of us, three domains, each owning an ingestion → features → model vertical slice:

| Person | Domain | Owns |
|---|---|---|
| _TBD_ | Transit | `ingestion/transit/`, `features/transit/`, `models/transit/` | Ben
| _TBD_ | Energy | `ingestion/energy/`, `features/energy/`, `models/energy/` | Nick
| _TBD_ | Weather & Anomalies | `ingestion/weather/`, `features/weather/`, `models/weather/` | Andrei 

Fill in names above. Shared/integration folders (`data/`, `serving/`, `frontend/`, `.github/workflows/`) are joint-owned — coordinate before restructuring them.

## What We're Building

See `URBAN_PULSE_PROJECT_PLAN.md` for the full architecture, phased timeline, and tech stack decision log. Short version: ingest city data (transit, energy, weather) → store in BigQuery/R2 → engineer features → train forecasting models → serve via FastAPI → display on a Next.js dashboard.

## Cross-Domain Dependencies

- The weather/anomalies model is a **producer** for the other two: anomaly flags and weather features from `features/weather/` are meant to be imported by `features/transit/` and `features/energy/`, not recomputed.
- `serving/` and `frontend/` are shared apps — each domain adds a router/view rather than forking the app. See `serving/README.md` and `frontend/README.md` for the convention.
- Data schemas live in `data/schemas/` — update the relevant schema file whenever a raw data source's shape changes so the other two people aren't surprised.

## Conventions

_Add to this section as we agree on things._

- Python: TBD (formatter/linter choice)
- Commit style: TBD
- Branch naming: TBD (suggest `<domain>/<short-description>`, e.g. `transit/lag-features`)
- Secrets: never commit tokens/keys — use `.env` locally (gitignored) and GitHub Actions secrets in CI

## Open Decisions / Questions

_Running list — add questions here as they come up, remove once resolved._

-

## Useful Commands

_Add project-specific commands here as they're established (e.g. how to run ingestion locally, how to run the API, how to run the frontend dev server)._
