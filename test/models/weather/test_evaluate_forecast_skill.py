import math

import pandas as pd
import pytest

from models.weather.evaluate_forecast_skill import (
    align_forecasts_with_observations,
    evaluate_forecast_skill,
)


def make_forecasts() -> pd.DataFrame:
    timestamps = pd.date_range("2026-09-01", periods=3, freq="h", tz="UTC")
    rows = []
    horizon_values = {
        24: {
            "temperature": [10.0, 12.0, 14.0],
            "apparent": [9.0, 11.0, 13.0],
            "precipitation": [0.0, 1.0, 0.0],
            "rain": [0.0, 1.0, 0.0],
            "snowfall": [0.0, 0.0, 1.0],
        },
        48: {
            "temperature": [9.0, 10.0, 11.0],
            "apparent": [8.0, 9.0, 10.0],
            "precipitation": [0.0, 0.0, 1.0],
            "rain": [0.0, 0.0, 1.0],
            "snowfall": [0.0, 0.0, 0.0],
        },
    }
    for lead_time, values in horizon_values.items():
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "timestamp_utc": timestamp,
                    "lead_time_hours": lead_time,
                    "forecast_temperature_2m_c": values["temperature"][index],
                    "forecast_apparent_temperature_c": values["apparent"][index],
                    "forecast_precipitation_mm": values["precipitation"][index],
                    "forecast_rain_mm": values["rain"][index],
                    "forecast_snowfall_cm": values["snowfall"][index],
                }
            )
    return pd.DataFrame(rows)


def make_observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2026-09-01", periods=3, freq="h", tz="UTC"
            ),
            "temperature_2m_c": [11.0, 11.0, 15.0],
            "apparent_temperature_c": [10.0, 10.0, 14.0],
            "precipitation_mm": [0.0, 1.0, 1.0],
            "rain_mm": [0.0, 1.0, 1.0],
            "snowfall_cm": [0.0, 0.0, 1.0],
        }
    )


def test_evaluate_forecast_skill_reports_metrics_by_horizon():
    metrics = evaluate_forecast_skill(make_forecasts(), make_observations())

    assert metrics["lead_time_hours"].tolist() == [24, 48]
    day_one = metrics.iloc[0]
    day_two = metrics.iloc[1]
    assert day_one["matched_hours"] == 3
    assert day_one["temperature_mae_c"] == pytest.approx(1.0)
    assert day_two["temperature_mae_c"] == pytest.approx(7 / 3)
    assert day_one["rain_precision"] == pytest.approx(1.0)
    assert day_one["rain_recall"] == pytest.approx(0.5)
    assert day_one["snow_precision"] == pytest.approx(1.0)
    assert day_one["snow_recall"] == pytest.approx(1.0)
    assert math.isnan(day_two["snow_precision"])
    assert day_two["snow_recall"] == pytest.approx(0.0)


def test_alignment_does_not_modify_inputs():
    forecasts = make_forecasts()
    observations = make_observations()
    original_forecasts = forecasts.copy(deep=True)
    original_observations = observations.copy(deep=True)

    align_forecasts_with_observations(forecasts, observations)

    pd.testing.assert_frame_equal(forecasts, original_forecasts)
    pd.testing.assert_frame_equal(observations, original_observations)


def test_alignment_rejects_duplicate_observations():
    observations = pd.concat(
        [make_observations(), make_observations().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate UTC timestamps"):
        align_forecasts_with_observations(make_forecasts(), observations)


def test_alignment_rejects_nonoverlapping_data():
    observations = make_observations()
    observations["timestamp_utc"] += pd.Timedelta(days=30)

    with pytest.raises(ValueError, match="no overlapping hours"):
        align_forecasts_with_observations(make_forecasts(), observations)


def test_negative_occurrence_threshold_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        evaluate_forecast_skill(
            make_forecasts(),
            make_observations(),
            rain_threshold_mm=-0.1,
        )

