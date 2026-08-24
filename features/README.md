# features/

Feature engineering scripts that turn `data/raw/` into `data/processed/`: time features, lags, rolling averages, joins against weather/events, etc.

One subfolder per domain, owned by that domain's teammate:

- `transit/`
- `energy/`
- `weather/`

If a feature is useful across domains (e.g. weather-derived features feeding the transit model), put the shared logic in `features/weather/` and import it from the consuming domain rather than duplicating it.
