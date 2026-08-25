from datetime import date
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from ingestion.weather.fetch_open_meteo_history import (
    HOURLY_VARIABLES,
    fetch_weather,
    response_to_dataframe,
    validate_hourly_data,
)


def make_hourly_payload(periods: int = 168) -> dict:
    """Build a valid Open-Meteo-shaped payload for use in tests."""
    timestamps = (
        pd.date_range(
            start="2024-01-01",
            periods=periods,
            freq="h",
            tz="UTC",
        )
        .strftime("%Y-%m-%dT%H:%M")
        .tolist()
    )

    return {
        "hourly": {
            "time": timestamps,
            "temperature_2m": [5.0] * periods,
            "relative_humidity_2m": [70.0] * periods,
            "apparent_temperature": [3.0] * periods,
            "precipitation": [0.0] * periods,
            "rain": [0.0] * periods,
            "snowfall": [0.0] * periods,
            "weather_code": [0] * periods,
            "wind_speed_10m": [12.0] * periods,
            "wind_gusts_10m": [20.0] * periods,
        }
    }


def make_mock_response(payload: dict) -> Mock:
    """Build a requests.Response mock whose JSON body is controlled locally."""
    response = Mock(spec=requests.Response)
    response.json.return_value = payload
    return response


def test_required_columns_exist():
    response = make_mock_response(make_hourly_payload())

    df = response_to_dataframe(response)

    expected_columns = {"timestamp_utc", *HOURLY_VARIABLES}
    assert expected_columns.issubset(df.columns)


def test_missing_required_column_raises_error():
    payload = make_hourly_payload()
    del payload["hourly"]["snowfall"]

    with pytest.raises(ValueError, match="missing hourly fields"):
        validate_hourly_data(payload["hourly"])


def test_timestamps_are_sorted():
    payload = make_hourly_payload()
    for values in payload["hourly"].values():
        values.reverse()

    response = make_mock_response(payload)
    df = response_to_dataframe(response)

    assert df["timestamp_utc"].is_monotonic_increasing


def test_duplicate_timestamps_raise_error():
    payload = make_hourly_payload()
    payload["hourly"]["time"][1] = payload["hourly"]["time"][0]
    response = make_mock_response(payload)

    with pytest.raises(ValueError, match="duplicate timestamps"):
        response_to_dataframe(response)


def test_seven_utc_days_produce_168_rows():
    response = make_mock_response(make_hourly_payload(periods=7 * 24))

    df = response_to_dataframe(response)

    assert len(df) == 168


@pytest.mark.parametrize("invalid_humidity", [-0.1, 100.1])
def test_humidity_outside_valid_range_raises_error(invalid_humidity):
    payload = make_hourly_payload()
    payload["hourly"]["relative_humidity_2m"][0] = invalid_humidity
    response = make_mock_response(payload)

    with pytest.raises(ValueError, match="relative_humidity_2m"):
        response_to_dataframe(response)


@pytest.mark.parametrize(
    "column",
    ["precipitation", "rain", "snowfall"],
)
def test_precipitation_fields_cannot_be_negative(column):
    payload = make_hourly_payload()
    payload["hourly"][column][0] = -0.1
    response = make_mock_response(payload)

    with pytest.raises(ValueError, match=column):
        response_to_dataframe(response)


def test_unequal_hourly_array_lengths_raise_error():
    payload = make_hourly_payload()
    payload["hourly"]["wind_speed_10m"].pop()

    with pytest.raises(
        ValueError,
        match="Hourly arrays have different lengths",
    ):
        validate_hourly_data(payload["hourly"])


@patch("ingestion.weather.fetch_open_meteo_history.requests.get")
def test_unsuccessful_http_response_raises_error(mock_get):
    response = Mock(spec=requests.Response)
    response.raise_for_status.side_effect = requests.HTTPError(
        "503 Service Unavailable"
    )
    mock_get.return_value = response

    with pytest.raises(requests.HTTPError):
        fetch_weather(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 7),
        )

    mock_get.assert_called_once()
