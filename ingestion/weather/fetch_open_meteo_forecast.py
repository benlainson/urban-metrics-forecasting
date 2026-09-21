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
    parser.add_argument(
        "--processed-archive-dir",
        type=Path,
        default=None,
        help="Optional directory for immutable normalized forecast runs.",
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


def add_forecast_provenance(
    forecast: pd.DataFrame,
    retrieved_at: datetime,
) -> pd.DataFrame:
    """Add retrieval and lead-time metadata and remove already-past rows.

    ``timestamp_utc`` remains the forecast valid time so the result stays
    compatible with the shared weather feature builder. Open-Meteo's daily
    forecast response begins at midnight UTC, so a daytime retrieval can
    contain hours that have already passed; those rows are not point-in-time
    forecasts and are excluded here.
    """
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("retrieved_at must be timezone-aware.")

    if not isinstance(forecast["timestamp_utc"].dtype, pd.DatetimeTZDtype):
        raise ValueError("timestamp_utc must be a timezone-aware datetime column.")

    retrieved_at_utc = pd.Timestamp(retrieved_at).tz_convert("UTC")
    forecast_run = forecast.copy(deep=True)
    forecast_run["timestamp_utc"] = forecast_run[
        "timestamp_utc"
    ].dt.tz_convert("UTC")
    forecast_run = forecast_run[
        forecast_run["timestamp_utc"] >= retrieved_at_utc
    ].copy()

    if forecast_run.empty:
        raise ValueError("Forecast response contains no future valid times.")

    forecast_run.insert(0, "retrieved_at_utc", retrieved_at_utc)
    forecast_run.insert(
        2,
        "lead_time_hours",
        (
            forecast_run["timestamp_utc"]
            - forecast_run["retrieved_at_utc"]
        ).dt.total_seconds()
        / 3600,
    )

    primary_key = [
        "retrieved_at_utc",
        "timestamp_utc",
        "latitude",
        "longitude",
    ]
    if forecast_run.duplicated(primary_key).any():
        raise ValueError("Forecast run contains duplicate primary keys.")
    if (forecast_run["lead_time_hours"] < 0).any():
        raise ValueError("Forecast run contains a negative lead time.")

    return forecast_run.reset_index(drop=True)


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


def save_processed_forecast(
    forecast_run: pd.DataFrame,
    latest_output_path: Path,
    archive_output_dir: Path,
    retrieved_at: datetime,
) -> Path:
    """Save both the latest forecast and a timestamped normalized run."""
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("retrieved_at must be timezone-aware.")

    timestamp = retrieved_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_output_path = (
        archive_output_dir
        / f"open_meteo_forecast_chicago_{timestamp}.parquet"
    )
    if archive_output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite archived forecast: {archive_output_path}"
        )

    save_processed_data(forecast_run, archive_output_path)
    save_processed_data(forecast_run, latest_output_path)
    return archive_output_path


def print_summary(
    df: pd.DataFrame,
    raw_output_path: Path,
    latest_output_path: Path,
    archive_output_path: Path,
) -> None:
    print(f"Saved raw forecast to: {raw_output_path}")
    print(f"Saved latest normalized forecast to: {latest_output_path}")
    print(f"Archived normalized forecast to: {archive_output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Retrieved at: {df['retrieved_at_utc'].iloc[0]}")
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
    processed_archive_dir = args.processed_archive_dir or (
        repo_root / "data" / "processed" / "weather" / "forecasts"
    )

    response = fetch_forecast(args.forecast_days)
    retrieved_at = datetime.now(timezone.utc)
    normalized_forecast = response_to_normalized_dataframe(response)
    forecast_run = add_forecast_provenance(
        normalized_forecast,
        retrieved_at,
    )

    raw_output_path = save_raw_response(
        response=response,
        output_dir=raw_output_dir,
        retrieved_at=retrieved_at,
    )
    archive_output_path = save_processed_forecast(
        forecast_run=forecast_run,
        latest_output_path=processed_output_path,
        archive_output_dir=processed_archive_dir,
        retrieved_at=retrieved_at,
    )
    print_summary(
        forecast_run,
        raw_output_path,
        processed_output_path,
        archive_output_path,
    )


if __name__ == "__main__":
    main()
