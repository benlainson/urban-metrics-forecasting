from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


COLUMN_NAMES = {
    "time": "timestamp_utc",
    "temperature_2m": "temperature_2m_c",
    "relative_humidity_2m": "relative_humidity_2m_pct",
    "apparent_temperature": "apparent_temperature_c",
    "precipitation": "precipitation_mm",
    "rain": "rain_mm",
    "snowfall": "snowfall_cm",
    "wind_speed_10m": "wind_speed_10m_kmh",
    "wind_gusts_10m": "wind_gusts_10m_kmh",
}

OUTPUT_COLUMNS = [
    "timestamp_utc",
    "temperature_2m_c",
    "relative_humidity_2m_pct",
    "apparent_temperature_c",
    "precipitation_mm",
    "rain_mm",
    "snowfall_cm",
    "weather_code",
    "wind_speed_10m_kmh",
    "wind_gusts_10m_kmh",
    "latitude",
    "longitude",
]

NONNEGATIVE_COLUMNS = [
    "precipitation_mm",
    "rain_mm",
    "snowfall_cm",
    "wind_speed_10m_kmh",
    "wind_gusts_10m_kmh",
]

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Normalize a raw Open-Meteo JSON response."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        type=Path,
        help="Path to a raw Open-Meteo JSON file.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional path for the processed Parquet file.",
    )
    return parser.parse_args()

def load_raw_response(input_file: Path) -> dict[str, any]:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    
    with input_file.open('r', encoding='utf-8') as file:
        payload = json.load(file)

    if "hourly" not in payload:
        raise ValueError("Raw response does not contain an 'hourly' object.")

    if "latitude" not in payload or "longitude" not in payload:
        raise ValueError("Raw response is missing latitude or longitude.")
    
    return payload

def normalize_weather(payload: dict[str, any]) -> pd.DataFrame:
    hourly = payload['hourly']

    if not isinstance(hourly, dict):
        raise ValueError("'hourly' must be an object.")

    lengths = {
        field: len(values)
        for field, values in hourly.items()
        if isinstance(values, list)
    }

    if len(lengths) != len(hourly):
        raise ValueError("Every hourly field must contain an array.")

    if len(set(lengths.values())) != 1:
        raise ValueError(f"Hourly arrays have different lengths: {lengths}")
    
    df = pd.DataFrame(hourly)
    df = df.rename(columns=COLUMN_NAMES)

    #change date value to datetime object
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="raise",
    )

    # Latitude and longitude occur once in the API response, so copy them
    # onto every hourly row.
    df["latitude"] = float(payload["latitude"])
    df["longitude"] = float(payload["longitude"])

    numeric_columns = [
        column
        for column in OUTPUT_COLUMNS
        if column not in {"timestamp_utc", "weather_code"}
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    # Pandas nullable integer type permits missing weather codes.
    df["weather_code"] = pd.to_numeric(
        df["weather_code"],
        errors="raise",
    ).astype("Int64")

    missing_columns = [
        column for column in OUTPUT_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Normalized data is missing columns: {missing_columns}"
        )

    df = df[OUTPUT_COLUMNS]
    df = df.sort_values(
        ["timestamp_utc", "latitude", "longitude"]
    ).reset_index(drop=True)

    validate_weather(df)
    return df


def validate_weather(df: pd.DataFrame) -> None:
    required_not_null = [
        "timestamp_utc",
        "latitude",
        "longitude",
    ]

    if df[required_not_null].isna().any().any():
        raise ValueError(
            "Timestamp, latitude, and longitude cannot be missing."
        )

    primary_key = [
        "timestamp_utc",
        "latitude",
        "longitude",
    ]

    if df.duplicated(primary_key).any():
        duplicate_count = int(df.duplicated(primary_key).sum())
        raise ValueError(
            f"Found {duplicate_count} duplicate primary keys."
        )

    humidity = df["relative_humidity_2m_pct"].dropna()
    if not humidity.between(0, 100).all():
        raise ValueError("Relative humidity must be between 0 and 100.")

    for column in NONNEGATIVE_COLUMNS:
        values = df[column].dropna()
        if (values < 0).any():
            raise ValueError(f"{column} cannot contain negative values.")

def save_processed_data(
    df: pd.DataFrame,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)


def print_summary(
    df: pd.DataFrame,
    output_file: Path,
) -> None:
    print(f"Saved processed data to: {output_file}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(
        f"Date range: {df['timestamp_utc'].min()} "
        f"to {df['timestamp_utc'].max()}"
    )

    print("\nColumn names:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nMissing values:")
    print(df.isna().sum().to_string())


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    output_file = args.output_file
    if output_file is None:
        output_file = (
            repo_root
            / "data"
            / "processed"
            / "weather"
            / f"{args.input_file.stem}.parquet"
        )

    payload = load_raw_response(args.input_file)
    df = normalize_weather(payload)
    save_processed_data(df, output_file)
    print_summary(df, output_file)


if __name__ == "__main__":
    main()