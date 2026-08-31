# ingestion/weather/

Owner: weather & anomalies model teammate.

Scripts that pull Open-Meteo (and any other weather/event data) used both as a standalone forecast target and as external features for the transit/energy models.

## Latest forecast

`fetch_open_meteo_forecast.py` downloads 1–16 days of hourly Chicago weather
from Open-Meteo's generic forecast endpoint. It archives the untouched response
under `data/raw/weather/` and writes the normalized latest forecast to
`data/processed/weather/open_meteo_forecast_latest.parquet`.

```bash
.venv/bin/python -m ingestion.weather.fetch_open_meteo_forecast \
  --forecast-days 7
```

The forecast requests the same variables, UTC timestamps, units, and normalized
column contract as the historical pipeline. A timestamp in each raw filename
records when that forecast was retrieved; the processed `latest` file is
replaced on each successful run for downstream inference.
