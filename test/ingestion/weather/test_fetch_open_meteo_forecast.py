import argparse
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from features.weather.normalize_open_meteo import OUTPUT_COLUMNS
from ingestion.weather.fetch_open_meteo_forecast import (
    FORECAST_URL,
    fetch_forecast,
    parse_forecast_days,
    response_to_normalized_dataframe,
    save_raw_response,
)
from ingestion.weather.fetch_open_meteo_history import HOURLY_VARIABLES


def make_forecast_payload(periods: int = 168) -> dict:
    timestamps = (
        pd.date_range(
            start="2026-08-30",
            periods=periods,
            freq="h",
            tz="UTC",
        )
        .strftime("%Y-%m-%dT%H:%M")
        .tolist()
    )
    return {
        "latitude": 41.875,
        "longitude": -87.625,
        "hourly": {
            "time": timestamps,
            "temperature_2m": [25.0] * periods,
            "relative_humidity_2m": [60.0] * periods,
            "apparent_temperature": [26.0] * periods,
            "precipitation": [0.0] * periods,
            "rain": [0.0] * periods,
            "snowfall": [0.0] * periods,
            "weather_code": [0] * periods,
            "wind_speed_10m": [10.0] * periods,
            "wind_gusts_10m": [18.0] * periods,
        },
    }


def make_mock_response(payload: dict) -> Mock:
    response = Mock(spec=requests.Response)
    response.json.return_value = payload
    response.content = b'{"untouched": true}'
    return response


@patch("ingestion.weather.fetch_open_meteo_forecast.requests.get")
def test_fetch_forecast_uses_shared_variables_and_utc(mock_get):
    response = make_mock_response(make_forecast_payload())
    mock_get.return_value = response

    result = fetch_forecast(forecast_days=7)

    assert result is response
    mock_get.assert_called_once()
    call = mock_get.call_args
    assert call.args[0] == FORECAST_URL
    assert call.kwargs["params"]["hourly"] == ",".join(HOURLY_VARIABLES)
    assert call.kwargs["params"]["timezone"] == "UTC"
    assert call.kwargs["params"]["forecast_days"] == 7
    response.raise_for_status.assert_called_once_with()


@pytest.mark.parametrize("value", ["1", "7", "16"])
def test_parse_forecast_days_accepts_supported_range(value):
    assert parse_forecast_days(value) == int(value)


@pytest.mark.parametrize("value", ["0", "17", "seven"])
def test_parse_forecast_days_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError, match="forecast-days"):
        parse_forecast_days(value)


def test_fetch_forecast_rejects_invalid_days_before_request():
    with pytest.raises(ValueError, match="forecast_days"):
        fetch_forecast(forecast_days=17)


def test_forecast_uses_the_normalized_historical_schema():
    response = make_mock_response(make_forecast_payload())

    result = response_to_normalized_dataframe(response)

    assert result.columns.tolist() == OUTPUT_COLUMNS
    assert len(result) == 168
    assert isinstance(result["timestamp_utc"].dtype, pd.DatetimeTZDtype)
    assert str(result["timestamp_utc"].dt.tz) == "UTC"
    assert result["weather_code"].dtype == "Int64"


def test_missing_hourly_variable_raises_error():
    payload = make_forecast_payload()
    del payload["hourly"]["snowfall"]
    response = make_mock_response(payload)

    with pytest.raises(ValueError, match="missing hourly fields"):
        response_to_normalized_dataframe(response)


def test_save_raw_response_preserves_exact_body_and_uses_retrieval_time(tmp_path):
    response = make_mock_response(make_forecast_payload())
    retrieved_at = datetime(2026, 8, 30, 15, 4, 5, tzinfo=timezone.utc)

    output_path = save_raw_response(response, tmp_path, retrieved_at)

    assert output_path.name == "open_meteo_forecast_chicago_20260830T150405Z.json"
    assert output_path.read_bytes() == response.content


def test_save_raw_response_requires_aware_retrieval_time(tmp_path):
    response = make_mock_response(make_forecast_payload())

    with pytest.raises(ValueError, match="timezone-aware"):
        save_raw_response(
            response,
            tmp_path,
            datetime(2026, 8, 30, 15, 4, 5),
        )


@patch("ingestion.weather.fetch_open_meteo_forecast.requests.get")
def test_http_error_is_propagated(mock_get):
    response = Mock(spec=requests.Response)
    response.raise_for_status.side_effect = requests.HTTPError(
        "503 Service Unavailable"
    )
    mock_get.return_value = response

    with pytest.raises(requests.HTTPError):
        fetch_forecast()
