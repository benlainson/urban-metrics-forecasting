import requests
import argparse
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "wind_speed_10m",
    "wind_gusts_10m",
]

CHICAGO_LATITUDE = 41.8781
CHICAGO_LONGITUDE = -87.6298

REQUEST_TIMEOUT_SECONDS = 30
NONNEGATIVE_HOURLY_VARIABLES = [
    "precipitation",
    "rain",
    "snowfall",
    "wind_speed_10m",
    "wind_gusts_10m",
]

def parse_date(value: str) -> date:
    """Parse an ISO-formatted date supplied on the command line."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}. Use YYYY-MM-DD."
        ) from exc

def parse_args() -> argparse.Namespace:
    """Take the args from the command line"""

    parser = argparse.ArgumentParser(
        description="Download historical hourly Chicago weather data."
    )
    parser.add_argument(
        "--start-date",
        required=True,
        type=parse_date,
        help="First date to download, formatted as YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        type=parse_date,
        help="Last date to download, formatted as YYYY-MM-DD.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional raw-data output directory.",
    )
    return parser.parse_args()

def validate_hourly_data(hourly: Any) -> dict[str, list[Any]]:
    """Ensure the API returned equally sized arrays for every hourly field."""
    if not isinstance(hourly, dict):
        raise ValueError("Response does not contain an 'hourly' object.")

    missing_variables = [
        variable
        for variable in ["time", *HOURLY_VARIABLES]
        if variable not in hourly
    ]
    if missing_variables:
        raise ValueError(
            f"Response is missing hourly fields: {missing_variables}"
        )

    invalid_fields = [
        name for name, values in hourly.items() if not isinstance(values, list)
    ]
    if invalid_fields:
        raise ValueError(
            f"These hourly fields are not arrays: {invalid_fields}"
        )

    lengths = {name: len(values) for name, values in hourly.items()}

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Hourly arrays have different lengths: {lengths}"
        )

    if lengths["time"] == 0:
        raise ValueError("Open-Meteo returned no hourly observations.")

    return hourly


def fetch_weather(start_date: date, end_date: date) -> requests.Response:
    """Request historical hourly weather data from Open-Meteo."""
    params = {
        "latitude": CHICAGO_LATITUDE,
        "longitude": CHICAGO_LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
    }

    response = requests.get(
        ARCHIVE_URL,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    return response


def response_to_dataframe(response: requests.Response) -> pd.DataFrame:
    """Validate an Open-Meteo response and convert it to a DataFrame."""
    payload = response.json()
    hourly = validate_hourly_data(payload.get("hourly"))

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "timestamp_utc"})

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="raise",
    )

    humidity = pd.to_numeric(
        df["relative_humidity_2m"], errors="raise"
    ).dropna()
    if not humidity.between(0, 100).all():
        raise ValueError("relative_humidity_2m must be between 0 and 100.")

    for column in NONNEGATIVE_HOURLY_VARIABLES:
        values = pd.to_numeric(df[column], errors="raise").dropna()
        if (values < 0).any():
            raise ValueError(f"{column} cannot contain negative values.")

    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    if df["timestamp_utc"].duplicated().any():
        duplicate_count = int(df["timestamp_utc"].duplicated().sum())
        raise ValueError(
            f"Found {duplicate_count} duplicate timestamps."
        )

    return df


def save_raw_response(
    response: requests.Response,
    output_dir: Path,
    start_date: date,
    end_date: date,
) -> Path:
    """Save the exact response body returned by Open-Meteo."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        "open_meteo_chicago_"
        f"{start_date.isoformat()}_"
        f"{end_date.isoformat()}.json"
    )

    output_path.write_bytes(response.content)
    return output_path


def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Print basic information about the downloaded data."""
    print(f"Saved raw response to: {output_path}")
    print(f"Rows: {len(df):,}")
    print(
        "Date range: "
        f"{df['timestamp_utc'].min()} to "
        f"{df['timestamp_utc'].max()}"
    )

    print("\nMissing values:")
    print(df.isna().sum().to_string())


def main() -> None:
    args = parse_args()

    if args.end_date < args.start_date:
        raise ValueError("--end-date cannot be earlier than --start-date.")

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or repo_root / "data" / "raw" / "weather"

    response = fetch_weather(
        start_date=args.start_date,
        end_date=args.end_date,
    )

    df = response_to_dataframe(response)

    output_path = save_raw_response(
        response=response,
        output_dir=output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print_summary(df, output_path)


if __name__ == "__main__":
    main()
