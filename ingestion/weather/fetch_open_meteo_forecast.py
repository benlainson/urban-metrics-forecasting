"""Fetch and normalize the latest hourly Open-Meteo forecast for Chicago."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from features.weather.normalize_open_meteo import (
    normalize_weather,
    save_processed_data,
)
from ingestion.weather.fetch_open_meteo_history import (
    CHICAGO_LATITUDE,
    CHICAGO_LONGITUDE,
    HOURLY_VARIABLES,
    REQUEST_TIMEOUT_SECONDS,
    validate_hourly_data,
)


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_FORECAST_DAYS = 7
MIN_FORECAST_DAYS = 1
MAX_FORECAST_DAYS = 16


def parse_forecast_days(value: str) -> int:
    """Parse and validate Open-Meteo's supported forecast-day range."""
    try:
        forecast_days = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--forecast-days must be an integer."
        ) from exc

    if not MIN_FORECAST_DAYS <= forecast_days <= MAX_FORECAST_DAYS:
        raise argparse.ArgumentTypeError(
            "--forecast-days must be between "
            f"{MIN_FORECAST_DAYS} and {MAX_FORECAST_DAYS}."
        )

    return forecast_days


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the latest hourly Chicago weather forecast."
    )
    parser.add_argument(
        "--forecast-days",
        type=parse_forecast_days,
        default=DEFAULT_FORECAST_DAYS,
        help="Number of forecast days to download (1-16; default: 7).",
    )
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        default=None,
        help="Optional directory for timestamped raw API responses.",
    )
    parser.add_argument(
        "--processed-output-file",
        type=Path,
        default=None,
        help="Optional path for the normalized latest-forecast Parquet file.",
    )
    return parser.parse_args()


def fetch_forecast(forecast_days: int = DEFAULT_FORECAST_DAYS) -> requests.Response:
    """Request the latest hourly forecast from Open-Meteo."""
    if not MIN_FORECAST_DAYS <= forecast_days <= MAX_FORECAST_DAYS:
        raise ValueError(
            f"forecast_days must be between {MIN_FORECAST_DAYS} "
            f"and {MAX_FORECAST_DAYS}."
        )

    params = {
        "latitude": CHICAGO_LATITUDE,
        "longitude": CHICAGO_LONGITUDE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
        "forecast_days": forecast_days,
    }
    response = requests.get(
        FORECAST_URL,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response


def response_to_normalized_dataframe(
    response: requests.Response,
) -> pd.DataFrame:
    """Validate and normalize a forecast using the shared weather contract."""
    payload = response.json()
    validate_hourly_data(payload.get("hourly"))
    return normalize_weather(payload)


def save_raw_response(
    response: requests.Response,
    output_dir: Path,
    retrieved_at: datetime,
) -> Path:
    """Archive the exact API response under a unique retrieval timestamp."""
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("retrieved_at must be timezone-aware.")

    retrieved_at_utc = retrieved_at.astimezone(timezone.utc)
    timestamp = retrieved_at_utc.strftime("%Y%m%dT%H%M%SZ")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"open_meteo_forecast_chicago_{timestamp}.json"
    output_path.write_bytes(response.content)
    return output_path


def print_summary(
    df: pd.DataFrame,
    raw_output_path: Path,
    processed_output_path: Path,
) -> None:
    print(f"Saved raw forecast to: {raw_output_path}")
    print(f"Saved normalized forecast to: {processed_output_path}")
    print(f"Rows: {len(df):,}")
    print(
        "Valid-time range: "
        f"{df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}"
    )
    print("\nMissing values:")
    print(df.isna().sum().to_string())


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    raw_output_dir = (
        args.raw_output_dir or repo_root / "data" / "raw" / "weather"
    )
    processed_output_path = args.processed_output_file or (
        repo_root
        / "data"
        / "processed"
        / "weather"
        / "open_meteo_forecast_latest.parquet"
    )

    retrieved_at = datetime.now(timezone.utc)
    response = fetch_forecast(args.forecast_days)
    forecast = response_to_normalized_dataframe(response)

    raw_output_path = save_raw_response(
        response=response,
        output_dir=raw_output_dir,
        retrieved_at=retrieved_at,
    )
    save_processed_data(forecast, processed_output_path)
    print_summary(forecast, raw_output_path, processed_output_path)


if __name__ == "__main__":
    main()

