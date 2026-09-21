import argparse
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from features.weather.normalize_open_meteo import OUTPUT_COLUMNS
from ingestion.weather.fetch_open_meteo_forecast import (
    FORECAST_URL,
    add_forecast_provenance,
    fetch_forecast,
    parse_forecast_days,
    response_to_normalized_dataframe,
    save_processed_forecast,
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


def test_forecast_provenance_removes_past_rows_and_calculates_lead_time():
    response = make_mock_response(make_forecast_payload())
    normalized = response_to_normalized_dataframe(response)
    retrieved_at = datetime(2026, 8, 30, 15, 30, tzinfo=timezone.utc)

    result = add_forecast_provenance(normalized, retrieved_at)

    assert result["timestamp_utc"].min() == pd.Timestamp(
        "2026-08-30T16:00:00Z"
    )
    assert result["retrieved_at_utc"].nunique() == 1
    assert result["retrieved_at_utc"].iloc[0] == pd.Timestamp(retrieved_at)
    assert result["lead_time_hours"].iloc[0] == pytest.approx(0.5)
    assert (result["timestamp_utc"] >= result["retrieved_at_utc"]).all()
    assert (result["lead_time_hours"] >= 0).all()


def test_forecast_provenance_does_not_modify_normalized_input():
    response = make_mock_response(make_forecast_payload())
    normalized = response_to_normalized_dataframe(response)
    original = normalized.copy(deep=True)

    add_forecast_provenance(
        normalized,
        datetime(2026, 8, 30, 15, 30, tzinfo=timezone.utc),
    )

    pd.testing.assert_frame_equal(normalized, original)


def test_forecast_provenance_requires_aware_retrieval_time():
    response = make_mock_response(make_forecast_payload())
    normalized = response_to_normalized_dataframe(response)

    with pytest.raises(ValueError, match="timezone-aware"):
        add_forecast_provenance(
            normalized,
            datetime(2026, 8, 30, 15, 30),
        )


def test_forecast_provenance_rejects_run_without_future_rows():
    response = make_mock_response(make_forecast_payload(periods=1))
    normalized = response_to_normalized_dataframe(response)

    with pytest.raises(ValueError, match="no future valid times"):
        add_forecast_provenance(
            normalized,
            datetime(2026, 8, 31, tzinfo=timezone.utc),
        )


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


@patch("ingestion.weather.fetch_open_meteo_forecast.save_processed_data")
def test_save_processed_forecast_writes_latest_and_timestamped_run(
    mock_save_processed_data,
    tmp_path,
):
    response = make_mock_response(make_forecast_payload())
    retrieved_at = datetime(2026, 8, 30, 15, 30, tzinfo=timezone.utc)
    normalized = response_to_normalized_dataframe(response)
    forecast_run = add_forecast_provenance(normalized, retrieved_at)
    latest_path = tmp_path / "latest.parquet"
    archive_dir = tmp_path / "forecasts"

    archive_path = save_processed_forecast(
        forecast_run,
        latest_path,
        archive_dir,
        retrieved_at,
    )

    assert archive_path.name == (
        "open_meteo_forecast_chicago_20260830T153000Z.parquet"
    )
    assert mock_save_processed_data.call_count == 2
    first_call, second_call = mock_save_processed_data.call_args_list
    pd.testing.assert_frame_equal(
        first_call.args[0],
        forecast_run,
    )
    assert first_call.args[1] == archive_path
    pd.testing.assert_frame_equal(second_call.args[0], forecast_run)
    assert second_call.args[1] == latest_path


def test_save_processed_forecast_refuses_to_overwrite_archive(tmp_path):
    response = make_mock_response(make_forecast_payload())
    retrieved_at = datetime(2026, 8, 30, 15, 30, tzinfo=timezone.utc)
    normalized = response_to_normalized_dataframe(response)
    forecast_run = add_forecast_provenance(normalized, retrieved_at)
    archive_dir = tmp_path / "forecasts"
    archive_dir.mkdir()
    archive_path = (
        archive_dir
        / "open_meteo_forecast_chicago_20260830T153000Z.parquet"
    )
    archive_path.write_bytes(b"existing archived run")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_processed_forecast(
            forecast_run,
            tmp_path / "latest.parquet",
            archive_dir,
            retrieved_at,
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
