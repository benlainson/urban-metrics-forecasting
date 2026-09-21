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
