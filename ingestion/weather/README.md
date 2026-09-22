# ingestion/weather/

Owner: weather & anomalies model teammate.

Scripts that pull Open-Meteo (and any other weather/event data) used both as a standalone forecast target and as external features for the transit/energy models.

## Latest forecast

`fetch_open_meteo_forecast.py` downloads 1–16 days of hourly Chicago weather
from Open-Meteo's generic forecast endpoint. It archives the untouched response
under `data/raw/weather/`, writes the normalized latest forecast to
`data/processed/weather/open_meteo_forecast_latest.parquet`, and preserves each
normalized run under `data/processed/weather/forecasts/`.

```bash
.venv/bin/python -m ingestion.weather.fetch_open_meteo_forecast \
  --forecast-days 7
```

The forecast requests the same variables, UTC timestamps, and units as the
historical pipeline. Forecast rows add `retrieved_at_utc` and
`lead_time_hours`; `timestamp_utc` remains the time for which the forecast is
valid. Hours earlier than the retrieval time are excluded so archived runs can
be used safely in backtests. The processed `latest` file is replaced on each
successful run for downstream inference, while timestamped run files remain
available for point-in-time evaluation.

## Historical 24/48-hour forecasts

`fetch_open_meteo_previous_runs.py` downloads forecasts aligned to fixed
24-hour and 48-hour lead times. These are separate from the continuously
updated live forecast archive and are intended for aggregate skill evaluation.

```bash
.venv/bin/python -m ingestion.weather.fetch_open_meteo_previous_runs \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

The processed output is a long table with one row per valid hour and horizon.
`forecast_reference_time_utc` is derived by subtracting the fixed horizon from
the valid time; it is a comparison reference, not a claim about the exact model
initialization timestamp.
