from datetime import date
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from ingestion.weather.fetch_open_meteo_previous_runs import (
    PREVIOUS_RUNS_URL,
    fetch_previous_runs,
    normalize_previous_runs,
    previous_run_variables,
)


def make_previous_runs_payload(periods: int = 3) -> dict:
    times = (
        pd.date_range("2026-09-01", periods=periods, freq="h", tz="UTC")
        .strftime("%Y-%m-%dT%H:%M")
        .tolist()
    )
    hourly = {"time": times}
    values = {
        "temperature_2m": [10.0, 12.0, 14.0],
        "apparent_temperature": [9.0, 11.0, 13.0],
        "precipitation": [0.0, 1.0, 0.0],
        "rain": [0.0, 1.0, 0.0],
        "snowfall": [0.0, 0.0, 1.0],
    }
    for horizon in (1, 2):
        for variable, variable_values in values.items():
            hourly[f"{variable}_previous_day{horizon}"] = [
                value + (horizon - 1) for value in variable_values[:periods]
            ]

    return {
        "latitude": 41.875,
        "longitude": -87.625,
        "hourly": hourly,
    }


@patch("ingestion.weather.fetch_open_meteo_previous_runs.requests.get")
def test_fetch_previous_runs_requests_24_and_48_hour_fields(mock_get):
    response = Mock(spec=requests.Response)
    mock_get.return_value = response

    result = fetch_previous_runs(date(2026, 9, 1), date(2026, 9, 2))

    assert result is response
    call = mock_get.call_args
    assert call.args[0] == PREVIOUS_RUNS_URL
    assert call.kwargs["params"]["start_date"] == "2026-09-01"
    assert call.kwargs["params"]["end_date"] == "2026-09-02"
    assert call.kwargs["params"]["timezone"] == "UTC"
    assert call.kwargs["params"]["temperature_unit"] == "celsius"
    assert call.kwargs["params"]["precipitation_unit"] == "mm"
    assert call.kwargs["params"]["hourly"] == ",".join(
        previous_run_variables()
    )
    response.raise_for_status.assert_called_once_with()


def test_fetch_previous_runs_rejects_reversed_dates():
    with pytest.raises(ValueError, match="end_date"):
        fetch_previous_runs(date(2026, 9, 2), date(2026, 9, 1))


def test_normalize_previous_runs_creates_long_fixed_horizon_table():
    result = normalize_previous_runs(make_previous_runs_payload())

    assert len(result) == 6
    assert result["lead_time_hours"].tolist() == [24, 48, 24, 48, 24, 48]
    assert set(result["lead_time_hours"]) == {24, 48}
    assert (
        result["timestamp_utc"] - result["forecast_reference_time_utc"]
    ).dt.total_seconds().div(3600).equals(
        result["lead_time_hours"].astype("float64")
    )
    assert result.loc[0, "forecast_temperature_2m_c"] == 10.0
    assert result.loc[1, "forecast_temperature_2m_c"] == 11.0


def test_normalize_previous_runs_allows_missing_forecast_values():
    payload = make_previous_runs_payload()
    payload["hourly"]["precipitation_previous_day1"][0] = None

    result = normalize_previous_runs(payload)

    assert pd.isna(result.loc[0, "forecast_precipitation_mm"])


def test_normalize_previous_runs_rejects_missing_field():
    payload = make_previous_runs_payload()
    del payload["hourly"]["snowfall_previous_day2"]

    with pytest.raises(ValueError, match="missing hourly fields"):
        normalize_previous_runs(payload)


def test_normalize_previous_runs_rejects_unequal_array_lengths():
    payload = make_previous_runs_payload()
    payload["hourly"]["rain_previous_day1"].pop()

    with pytest.raises(ValueError, match="different lengths"):
        normalize_previous_runs(payload)
