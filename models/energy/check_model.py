"""Independently reload an energy run and verify its held-out predictions/metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from models.energy.train_xgboost import BASELINES, DEFAULT_OUTPUT, FEATURE_COLUMNS, regression_metrics


def check_run(output: Path) -> None:
    run = json.loads((output / "metrics.json").read_text())
    features = pd.read_parquet(output / "test_features.parquet")
    predictions = pd.read_csv(output / "test_predictions.csv")
    periods = pd.to_datetime(predictions.period, utc=True)
    origins = pd.to_datetime(predictions.forecast_origin_utc, utc=True)
    if run["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Saved feature list does not match the demand-only model contract.")
    pd.testing.assert_series_equal(features.period.reset_index(drop=True), periods, check_names=False, check_dtype=False)
    if not periods.is_unique or not periods.is_monotonic_increasing:
        raise ValueError("Test timestamps are duplicated or unordered.")
    info = run["splits"]["test"]
    if len(predictions) != info["rows"] or not (
        (periods >= pd.Timestamp(info["start_inclusive"])) &
        (periods < pd.Timestamp(info["end_exclusive"]))
    ).all():
        raise ValueError("Test rows do not match the saved split boundaries.")
    if not (periods - origins).eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Forecast origins are not one hour before their targets.")
    model = XGBRegressor()
    model.load_model(output / "model.json")
    np.testing.assert_allclose(model.predict(features[FEATURE_COLUMNS]), predictions.xgboost, rtol=0, atol=1e-5)
    for name, column in BASELINES.items():
        np.testing.assert_allclose(features[column], predictions[name], rtol=0, atol=1e-5)
    for name in ["xgboost", *BASELINES]:
        scores = regression_metrics(predictions.actual, predictions[name])
        for metric, value in scores.items():
            np.testing.assert_allclose(value, run["scores"]["test"][name][metric], rtol=1e-8)
    print(f"PASS: {len(predictions):,} held-out predictions reproduced from the saved model.")
    print("PASS: hourly origins, chronological test boundaries, three baselines, MAE and RMSE verified.")
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    check_run(parser.parse_args().output)
