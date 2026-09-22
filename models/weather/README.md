# models/weather/

Owner: weather & anomalies model teammate.

Weather forecasting models and anomaly detection (e.g. IsolationForest over rolling metrics) used to flag unusual events — COVID-style drops, storms, holidays — that the transit and energy models can consume as a feature or explanatory overlay.

## Forecast skill evaluation

`evaluate_forecast_skill.py` joins fixed-horizon forecasts to normalized
observations using their valid UTC hour and reports metrics separately for each
lead time:

- Temperature and apparent-temperature MAE
- Precipitation and snowfall MAE
- Rain and snow occurrence precision and recall

```bash
.venv/bin/python -m models.weather.evaluate_forecast_skill \
  --forecast-file data/processed/weather/forecast_evaluation/open_meteo_previous_runs_chicago_2024-01-01_2024-12-31.parquet \
  --observed-file data/processed/weather/open_meteo_chicago_2024-01-01_2024-12-31.parquet \
  --output-file data/processed/weather/forecast_evaluation/weather_forecast_skill_2024.csv
```

Use `notebooks/weather_02_forecast_evaluation.ipynb` for interactive inspection;
keep metric calculations in the tested Python module.
