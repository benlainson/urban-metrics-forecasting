"""Fetch fixed-horizon historical forecasts for weather skill evaluation."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from features.weather.normalize_open_meteo import save_processed_data
from ingestion.weather.fetch_open_meteo_history import (
    CHICAGO_LATITUDE,
    CHICAGO_LONGITUDE,
    REQUEST_TIMEOUT_SECONDS,
    parse_date,
)


PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_HORIZON_DAYS = (1, 2)
EVALUATION_VARIABLES = (
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
)

FORECAST_COLUMN_NAMES = {
    "temperature_2m": "forecast_temperature_2m_c",
    "apparent_temperature": "forecast_apparent_temperature_c",
    "precipitation": "forecast_precipitation_mm",
    "rain": "forecast_rain_mm",
    "snowfall": "forecast_snowfall_cm",
}


def previous_run_variables() -> list[str]:
    """Return Open-Meteo fields for the 24- and 48-hour horizons."""
    return [
        f"{variable}_previous_day{horizon_days}"
        for horizon_days in FORECAST_HORIZON_DAYS
        for variable in EVALUATION_VARIABLES
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Chicago forecasts made 24 and 48 hours before valid time."
        )
    )
    parser.add_argument("--start-date", required=True, type=parse_date)
    parser.add_argument("--end-date", required=True, type=parse_date)
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        default=None,
        help="Optional directory for the untouched API response.",
    )
    parser.add_argument(
        "--processed-output-file",
        type=Path,
        default=None,
        help="Optional path for the normalized fixed-horizon forecasts.",
    )
    return parser.parse_args()


def fetch_previous_runs(start_date: date, end_date: date) -> requests.Response:
    """Request fixed 24/48-hour forecasts for an inclusive valid-time range."""
    if end_date < start_date:
        raise ValueError("end_date cannot be earlier than start_date.")

    params = {
        "latitude": CHICAGO_LATITUDE,
        "longitude": CHICAGO_LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(previous_run_variables()),
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }
    response = requests.get(
        PREVIOUS_RUNS_URL,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response


def _validate_hourly_payload(hourly: Any) -> dict[str, list[Any]]:
    if not isinstance(hourly, dict):
        raise ValueError("Response does not contain an 'hourly' object.")

    required_fields = ["time", *previous_run_variables()]
    missing_fields = [field for field in required_fields if field not in hourly]
    if missing_fields:
        raise ValueError(f"Response is missing hourly fields: {missing_fields}")

    invalid_fields = [
        field for field in required_fields if not isinstance(hourly[field], list)
    ]
    if invalid_fields:
        raise ValueError(f"Hourly fields must be arrays: {invalid_fields}")

    lengths = {field: len(hourly[field]) for field in required_fields}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Hourly arrays have different lengths: {lengths}")
    if lengths["time"] == 0:
        raise ValueError("Open-Meteo returned no previous-run forecasts.")

    return hourly


def normalize_previous_runs(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert suffixed previous-run fields into one row per horizon and hour."""
    if "latitude" not in payload or "longitude" not in payload:
        raise ValueError("Response is missing latitude or longitude.")

    hourly = _validate_hourly_payload(payload.get("hourly"))
    timestamps = pd.to_datetime(hourly["time"], utc=True, errors="raise")
    frames: list[pd.DataFrame] = []

    for horizon_days in FORECAST_HORIZON_DAYS:
        horizon = pd.DataFrame(
            {
                "timestamp_utc": timestamps,
                "lead_time_hours": horizon_days * 24,
                "latitude": float(payload["latitude"]),
                "longitude": float(payload["longitude"]),
            }
        )
        horizon["forecast_reference_time_utc"] = (
            horizon["timestamp_utc"] - pd.Timedelta(days=horizon_days)
        )

        for source_name, output_name in FORECAST_COLUMN_NAMES.items():
            api_field = f"{source_name}_previous_day{horizon_days}"
            horizon[output_name] = pd.to_numeric(
                hourly[api_field], errors="raise"
            )

        frames.append(horizon)

    forecasts = pd.concat(frames, ignore_index=True)
    forecasts = forecasts.sort_values(
        ["timestamp_utc", "lead_time_hours"]
    ).reset_index(drop=True)

    primary_key = [
        "timestamp_utc",
        "lead_time_hours",
        "latitude",
        "longitude",
    ]
    if forecasts.duplicated(primary_key).any():
        raise ValueError("Previous-run forecasts contain duplicate primary keys.")
    if (forecasts["lead_time_hours"] <= 0).any():
        raise ValueError("Previous-run lead times must be positive.")
    if not (
        forecasts["forecast_reference_time_utc"] < forecasts["timestamp_utc"]
    ).all():
        raise ValueError("Forecast reference times must precede valid times.")

    return forecasts


def save_raw_response(
    response: requests.Response,
    output_dir: Path,
    start_date: date,
    end_date: date,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        "open_meteo_previous_runs_chicago_"
        f"{start_date.isoformat()}_{end_date.isoformat()}.json"
    )
    output_path.write_bytes(response.content)
    return output_path


def main() -> None:
    args = parse_args()
    if args.end_date < args.start_date:
        raise ValueError("--end-date cannot be earlier than --start-date.")

    repo_root = Path(__file__).resolve().parents[2]
    raw_output_dir = (
        args.raw_output_dir or repo_root / "data" / "raw" / "weather"
    )
    processed_output_file = args.processed_output_file or (
        repo_root
        / "data"
        / "processed"
        / "weather"
        / "forecast_evaluation"
        / (
            "open_meteo_previous_runs_chicago_"
            f"{args.start_date.isoformat()}_{args.end_date.isoformat()}.parquet"
        )
    )

    response = fetch_previous_runs(args.start_date, args.end_date)
    forecasts = normalize_previous_runs(response.json())
    raw_path = save_raw_response(
        response,
        raw_output_dir,
        args.start_date,
        args.end_date,
    )
    save_processed_data(forecasts, processed_output_file)

    print(f"Saved raw previous runs to: {raw_path}")
    print(f"Saved normalized previous runs to: {processed_output_file}")
    print(f"Rows: {len(forecasts):,}")
    print(f"Horizons: {sorted(forecasts['lead_time_hours'].unique())}")


if __name__ == "__main__":
    main()
