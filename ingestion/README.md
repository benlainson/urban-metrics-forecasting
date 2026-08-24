# ingestion/

Scripts that pull data from external APIs and write it into `data/raw/` (and/or push to BigQuery — see `URBAN_PULSE_PROJECT_PLAN.md`). Automated on a schedule via `.github/workflows/`.

One subfolder per domain, owned by that domain's teammate:

- `transit/` — CTA bus/train tracker + historical ridership APIs
- `energy/` — EIA + utility APIs
- `weather/` — Open-Meteo API

Keep API tokens/secrets out of code — read them from environment variables (GitHub Actions secrets in CI, `.env` locally, already gitignored).
