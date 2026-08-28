"""Build reusable, leakage-safe features from normalized hourly weather data."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


LOCAL_TIMEZONE = "America/Chicago"
LOCATION_COLUMNS = ["latitude", "longitude"]
REQUIRED_COLUMNS = [
    "timestamp_utc",
    "temperature_2m_c",
    "precipitation_mm",
    "rain_mm",
    "snowfall_cm",
    *LOCATION_COLUMNS,
]

FEATURE_COLUMNS = [
    "local_hour",
    "local_day_of_week",
    "local_month",
    "temperature_change_1h_c",
    "precipitation_3h_mm",
    "precipitation_24h_mm",
    "snowfall_24h_cm",
    "is_raining",
    "is_snowing",
    "is_freezing",
]


@dataclass(frozen=True)
class WeatherFeatureThresholds:
    """Thresholds used to convert continuous weather values into flags."""

    rain_mm: float = 0.0
    snowfall_cm: float = 0.0
    freezing_c: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "rain_mm": self.rain_mm,
            "snowfall_cm": self.snowfall_cm,
            "freezing_c": self.freezing_c,
        }
        invalid = [
            name
            for name, value in values.items()
            if not isinstance(value, (int, float)) or not math.isfinite(value)
        ]
        if invalid:
            raise ValueError(f"Thresholds must be finite numbers: {invalid}")

        if self.rain_mm < 0 or self.snowfall_cm < 0:
            raise ValueError("Rain and snowfall thresholds cannot be negative.")


def _validate_input(df: pd.DataFrame) -> None:
    """Validate the parts of the normalized weather contract used here."""
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Weather data is missing required columns: {missing_columns}"
        )

    if not isinstance(df["timestamp_utc"].dtype, pd.DatetimeTZDtype):
        raise ValueError("timestamp_utc must be a timezone-aware datetime column.")

    if df[["timestamp_utc", *LOCATION_COLUMNS]].isna().any().any():
        raise ValueError("Timestamp, latitude, and longitude cannot be missing.")

    primary_key = ["timestamp_utc", *LOCATION_COLUMNS]
    duplicate_count = int(df.duplicated(primary_key).sum())
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate weather observation keys."
        )


def _nullable_flag(values: pd.Series, threshold: float) -> pd.Series:
    """Compare values to a threshold without treating missing data as false."""
    return values.gt(threshold).mask(values.isna()).astype("boolean")


def _add_lookback_features(group: pd.DataFrame) -> pd.DataFrame:
    """Add past-and-current features for one weather location."""
    group = group.sort_values("timestamp_utc").copy()
    elapsed = group["timestamp_utc"].diff()

    group["temperature_change_1h_c"] = (
        group["temperature_2m_c"]
        .diff()
        .where(elapsed.eq(pd.Timedelta(hours=1)))
    )

    indexed = group.set_index("timestamp_utc")
    group["precipitation_3h_mm"] = (
        indexed["precipitation_mm"]
        .rolling("3h", min_periods=3)
        .sum()
        .to_numpy()
    )
    group["precipitation_24h_mm"] = (
        indexed["precipitation_mm"]
        .rolling("24h", min_periods=24)
        .sum()
        .to_numpy()
    )
    group["snowfall_24h_cm"] = (
        indexed["snowfall_cm"]
        .rolling("24h", min_periods=24)
        .sum()
        .to_numpy()
    )

    return group


def build_weather_features(
    df: pd.DataFrame,
    thresholds: WeatherFeatureThresholds | None = None,
) -> pd.DataFrame:
    """Return normalized weather data with shared model features appended.

    Rolling values contain the current observation and earlier observations
    only. A full window of non-null hourly values is required, preventing a
    gap in the source data from masquerading as a complete accumulation.
    The input DataFrame is never modified.
    """
    _validate_input(df)
    thresholds = thresholds or WeatherFeatureThresholds()

    weather = df.copy(deep=True)
    weather["timestamp_utc"] = weather["timestamp_utc"].dt.tz_convert("UTC")
    weather = weather.sort_values(
        [*LOCATION_COLUMNS, "timestamp_utc"]
    ).reset_index(drop=True)

    location_groups = weather.groupby(
        LOCATION_COLUMNS,
        sort=False,
        dropna=False,
        group_keys=False,
    )
    featured = pd.concat(
        [_add_lookback_features(group) for _, group in location_groups],
        ignore_index=True,
    )

    local_timestamp = featured["timestamp_utc"].dt.tz_convert(LOCAL_TIMEZONE)
    featured["local_hour"] = local_timestamp.dt.hour.astype("int8")
    featured["local_day_of_week"] = local_timestamp.dt.dayofweek.astype("int8")
    featured["local_month"] = local_timestamp.dt.month.astype("int8")

    featured["is_raining"] = _nullable_flag(
        featured["rain_mm"], thresholds.rain_mm
    )
    featured["is_snowing"] = _nullable_flag(
        featured["snowfall_cm"], thresholds.snowfall_cm
    )
    featured["is_freezing"] = (
        featured["temperature_2m_c"]
        .le(thresholds.freezing_c)
        .mask(featured["temperature_2m_c"].isna())
        .astype("boolean")
    )

    return featured.sort_values(
        ["timestamp_utc", *LOCATION_COLUMNS]
    ).reset_index(drop=True)

