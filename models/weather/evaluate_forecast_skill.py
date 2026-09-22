"""Evaluate fixed-horizon weather forecasts against observed weather."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FORECAST_REQUIRED_COLUMNS = [
    "timestamp_utc",
    "lead_time_hours",
    "forecast_temperature_2m_c",
    "forecast_apparent_temperature_c",
    "forecast_precipitation_mm",
    "forecast_rain_mm",
    "forecast_snowfall_cm",
]
OBSERVED_REQUIRED_COLUMNS = [
    "timestamp_utc",
    "temperature_2m_c",
    "apparent_temperature_c",
    "precipitation_mm",
    "rain_mm",
    "snowfall_cm",
]


def _require_columns(
    df: pd.DataFrame,
    required: list[str],
    dataset_name: str,
) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing columns: {missing}")


def align_forecasts_with_observations(
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Join forecasts to observations by valid UTC hour."""
    _require_columns(forecasts, FORECAST_REQUIRED_COLUMNS, "Forecast data")
    _require_columns(observations, OBSERVED_REQUIRED_COLUMNS, "Observed data")

    forecast_data = forecasts.copy(deep=True)
    observed_data = observations.copy(deep=True)
    forecast_data["timestamp_utc"] = pd.to_datetime(
        forecast_data["timestamp_utc"], utc=True, errors="raise"
    )
    observed_data["timestamp_utc"] = pd.to_datetime(
        observed_data["timestamp_utc"], utc=True, errors="raise"
    )

    if observed_data["timestamp_utc"].duplicated().any():
        raise ValueError("Observed data contains duplicate UTC timestamps.")

    observed_values = observed_data[OBSERVED_REQUIRED_COLUMNS]
    aligned = forecast_data.merge(
        observed_values,
        on="timestamp_utc",
        how="inner",
        validate="many_to_one",
    )
    if aligned.empty:
        raise ValueError("Forecasts and observations have no overlapping hours.")

    return aligned.sort_values(
        ["lead_time_hours", "timestamp_utc"]
    ).reset_index(drop=True)


def _mean_absolute_error(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    paired = pd.concat([actual, predicted], axis=1).dropna()
    if paired.empty:
        return float("nan")
    return float((paired.iloc[:, 0] - paired.iloc[:, 1]).abs().mean())


def _occurrence_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    threshold: float,
) -> tuple[float, float]:
    paired = pd.concat([actual, predicted], axis=1).dropna()
    if paired.empty:
        return float("nan"), float("nan")

    actual_event = paired.iloc[:, 0] > threshold
    predicted_event = paired.iloc[:, 1] > threshold
    true_positive = int((actual_event & predicted_event).sum())
    predicted_positive = int(predicted_event.sum())
    actual_positive = int(actual_event.sum())

    precision = (
        true_positive / predicted_positive
        if predicted_positive
        else float("nan")
    )
    recall = (
        true_positive / actual_positive if actual_positive else float("nan")
    )
    return precision, recall


def evaluate_forecast_skill(
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    rain_threshold_mm: float = 0.1,
    snow_threshold_cm: float = 0.1,
) -> pd.DataFrame:
    """Calculate forecast error and event skill for each lead-time horizon."""
    if rain_threshold_mm < 0 or snow_threshold_cm < 0:
        raise ValueError("Occurrence thresholds cannot be negative.")

    aligned = align_forecasts_with_observations(forecasts, observations)
    rows: list[dict[str, float | int]] = []

    for lead_time, group in aligned.groupby("lead_time_hours", sort=True):
        rain_precision, rain_recall = _occurrence_metrics(
            group["rain_mm"],
            group["forecast_rain_mm"],
            rain_threshold_mm,
        )
        snow_precision, snow_recall = _occurrence_metrics(
            group["snowfall_cm"],
            group["forecast_snowfall_cm"],
            snow_threshold_cm,
        )
        rows.append(
            {
                "lead_time_hours": int(lead_time),
                "matched_hours": len(group),
                "temperature_mae_c": _mean_absolute_error(
                    group["temperature_2m_c"],
                    group["forecast_temperature_2m_c"],
                ),
                "apparent_temperature_mae_c": _mean_absolute_error(
                    group["apparent_temperature_c"],
                    group["forecast_apparent_temperature_c"],
                ),
                "precipitation_mae_mm": _mean_absolute_error(
                    group["precipitation_mm"],
                    group["forecast_precipitation_mm"],
                ),
                "snowfall_mae_cm": _mean_absolute_error(
                    group["snowfall_cm"],
                    group["forecast_snowfall_cm"],
                ),
                "rain_precision": rain_precision,
                "rain_recall": rain_recall,
                "snow_precision": snow_precision,
                "snow_recall": snow_recall,
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-horizon weather forecasts."
    )
    parser.add_argument("--forecast-file", required=True, type=Path)
    parser.add_argument("--observed-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forecasts = pd.read_parquet(args.forecast_file)
    observations = pd.read_parquet(args.observed_file)
    metrics = evaluate_forecast_skill(forecasts, observations)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_file, index=False)
    print(metrics.to_string(index=False))
    print(f"Saved metrics to: {args.output_file}")


if __name__ == "__main__":
    main()

