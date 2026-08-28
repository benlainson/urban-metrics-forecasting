import pandas as pd
import pytest

from features.weather.build_features import (
    WeatherFeatureThresholds,
    build_weather_features,
)


def make_weather_data(periods: int = 24) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2024-01-01", periods=periods, freq="h", tz="UTC"
            ),
            "temperature_2m_c": [float(value) for value in range(periods)],
            "precipitation_mm": [1.0] * periods,
            "rain_mm": [0.0] * periods,
            "snowfall_cm": [0.0] * periods,
            "latitude": [41.8781] * periods,
            "longitude": [-87.6298] * periods,
        }
    )


def test_builds_calendar_change_rolling_and_flag_features():
    df = make_weather_data()
    df.loc[0, "rain_mm"] = 0.5
    df.loc[1, "snowfall_cm"] = 0.2
    df.loc[0, "temperature_2m_c"] = -1.0

    result = build_weather_features(df)

    assert result.loc[0, "local_hour"] == 18
    assert result.loc[0, "local_day_of_week"] == 6
    assert result.loc[0, "local_month"] == 12
    assert pd.isna(result.loc[0, "temperature_change_1h_c"])
    assert result.loc[1, "temperature_change_1h_c"] == 2.0
    assert result.loc[2, "precipitation_3h_mm"] == 3.0
    assert result.loc[23, "precipitation_24h_mm"] == 24.0
    assert result.loc[23, "snowfall_24h_cm"] == pytest.approx(0.2)
    assert bool(result.loc[0, "is_raining"])
    assert bool(result.loc[1, "is_snowing"])
    assert bool(result.loc[0, "is_freezing"])


def test_does_not_modify_input_dataframe():
    df = make_weather_data(periods=3)
    original = df.copy(deep=True)

    build_weather_features(df)

    pd.testing.assert_frame_equal(df, original)


def test_locations_are_featured_independently():
    first = make_weather_data(periods=3)
    second = make_weather_data(periods=3)
    second["latitude"] = 42.0
    second["temperature_2m_c"] = [100.0, 110.0, 120.0]

    result = build_weather_features(pd.concat([first, second], ignore_index=True))
    second_location = result[result["latitude"] == 42.0].reset_index(drop=True)

    assert pd.isna(second_location.loc[0, "temperature_change_1h_c"])
    assert second_location.loc[1, "temperature_change_1h_c"] == 10.0
    assert second_location.loc[2, "precipitation_3h_mm"] == 3.0


def test_gap_does_not_create_a_one_hour_temperature_change():
    df = make_weather_data(periods=3).drop(index=1).reset_index(drop=True)

    result = build_weather_features(df)

    assert pd.isna(result.loc[1, "temperature_change_1h_c"])
    assert pd.isna(result.loc[1, "precipitation_3h_mm"])


def test_future_values_do_not_change_earlier_features():
    original = make_weather_data(periods=4)
    changed_future = original.copy(deep=True)
    changed_future.loc[3, "temperature_2m_c"] = 999.0
    changed_future.loc[3, "precipitation_mm"] = 999.0

    original_result = build_weather_features(original)
    changed_result = build_weather_features(changed_future)

    columns = [
        "temperature_change_1h_c",
        "precipitation_3h_mm",
    ]
    pd.testing.assert_frame_equal(
        original_result.loc[:2, columns],
        changed_result.loc[:2, columns],
    )


def test_missing_measurement_produces_missing_flag():
    df = make_weather_data(periods=1)
    df.loc[0, "rain_mm"] = None

    result = build_weather_features(df)

    assert pd.isna(result.loc[0, "is_raining"])


def test_custom_thresholds_are_applied():
    df = make_weather_data(periods=1)
    df.loc[0, "rain_mm"] = 0.5
    df.loc[0, "snowfall_cm"] = 0.5
    df.loc[0, "temperature_2m_c"] = -2.0
    thresholds = WeatherFeatureThresholds(
        rain_mm=1.0,
        snowfall_cm=1.0,
        freezing_c=-5.0,
    )

    result = build_weather_features(df, thresholds=thresholds)

    assert not bool(result.loc[0, "is_raining"])
    assert not bool(result.loc[0, "is_snowing"])
    assert not bool(result.loc[0, "is_freezing"])


def test_duplicate_observation_keys_raise_error():
    df = make_weather_data(periods=2)
    df.loc[1, "timestamp_utc"] = df.loc[0, "timestamp_utc"]

    with pytest.raises(ValueError, match="duplicate weather observation keys"):
        build_weather_features(df)


def test_naive_timestamps_raise_error():
    df = make_weather_data(periods=1)
    df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize(None)

    with pytest.raises(ValueError, match="timezone-aware"):
        build_weather_features(df)


def test_invalid_thresholds_raise_error():
    with pytest.raises(ValueError, match="cannot be negative"):
        WeatherFeatureThresholds(rain_mm=-0.1)
