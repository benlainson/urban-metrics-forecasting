# features/weather/

Owner: weather & anomalies model teammate.

Feature engineering for weather/anomaly detection, and the shared weather features (temperature, precipitation, snowfall, etc.) that the transit and energy models join against.

## Shared hourly features

`build_features.py` adds model-ready features to the normalized hourly weather
contract without modifying the input DataFrame. Transit and energy code should
import `build_weather_features` rather than recreate these calculations.

```python
from features.weather.build_features import build_weather_features

weather_features = build_weather_features(normalized_weather)
```

The output retains every normalized source column and adds:

- Chicago-local `local_hour`, `local_day_of_week`, and `local_month`
- `temperature_change_1h_c`
- `precipitation_3h_mm` and `precipitation_24h_mm`
- `snowfall_24h_cm`
- Nullable Boolean `is_raining`, `is_snowing`, and `is_freezing` flags

Rolling totals include the current observation and past observations only. A
complete window is required, so the first 2/23 hours and windows containing
missing measurements remain null. Calculations are isolated by latitude and
longitude to prevent values from one weather location leaking into another.

Flag thresholds can be changed through `WeatherFeatureThresholds`; the defaults
mean any rain or snowfall greater than zero and temperature at or below 0 °C.
