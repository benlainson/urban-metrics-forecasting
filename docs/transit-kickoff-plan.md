# Transit Kickoff Plan

Owner: transit model teammate. Checklist to get from zero to a working baseline model. Check items off as you go; update if scope changes.

## 0. Shared setup (blocks everyone, do once)

- [ ] Add root `requirements.txt` (pandas, requests, xgboost, prophet, wandb, google-cloud-bigquery, pyarrow, python-dotenv)
- [ ] Add `.env.example` documenting expected env vars (`CTA_APP_TOKEN`, `CTA_BUS_TRACKER_KEY`, `CTA_TRAIN_TRACKER_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, `WANDB_API_KEY`)
- [ ] Create the shared GCP project + BigQuery dataset `urban_pulse`, grant IAM to teammates (see root `CLAUDE.md`)

## 1. Register transit API access

Three separate registrations — not one, despite how `URBAN_PULSE_PROJECT_PLAN.md` groups them:

- [ ] Socrata app token — data.cityofchicago.org (historical ridership, removes rate limit)
- [ ] CTA Bus Tracker (BusTime) developer key — transitchicago.com/developers/bustracker
- [ ] CTA Train Tracker developer key — transitchicago.com/developers/traintracker

## 2. First data pull

- [ ] `ingestion/transit/fetch_cta_ridership.py` — pull daily boarding totals (`t2rn-p8d7`) + L station entries (`5neh-572f`) via Socrata, save to `data/raw/transit/`
- [ ] Run it manually, inspect shape/dtypes/date range, note anomalies (COVID drop, etc.)
- [ ] `ingestion/transit/push_to_bigquery.py` — load into `urban_pulse.transit_ridership_raw`

## 3. Automate

- [ ] `.github/workflows/ingest_transit.yml` — daily cron, uses `CTA_APP_TOKEN` secret

## 4. EDA

- [ ] `notebooks/transit_01_eda_ridership.ipynb` — date range, top stations, weekly/yearly seasonality, anomalies

## 5. Feature engineering

- [ ] `features/transit/engineer.py` — calendar features, lag features (1/7/14-day), rolling averages (7/30-day), write to `data/processed/`

## 6. Baseline models

- [ ] `models/transit/train_prophet.py` — Prophet baseline per station, log MAE to W&B
- [ ] `models/transit/train_xgboost.py` — XGBoost, beat Prophet, log to W&B, save model artifact

## Not in scope yet

Weather join, TFT, serving endpoints, frontend view — these depend on the weather teammate's `features/weather/` output and cross-team coordination. Revisit after step 6.
