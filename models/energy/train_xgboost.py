"""Train and evaluate a rolling one-hour-ahead energy demand forecast.

Run from the repository root: python -m models.energy.train_xgboost
Each row predicts demand at t with calendar information for t and demand
observed through t-1. Validation selects the tree count; test never fits it.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from features.energy.engineer import DEFAULT_MAX_DEMAND, build_features


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/raw/energy/eia_demand_pjm.parquet"
DEFAULT_OUTPUT = ROOT / "data/processed/energy/xgboost"
FEATURE_COLUMNS = [
    "hour", "dayofweek", "month", "year", "quarter", "is_weekend", "week_of_year",
    "demand_lag_1", "demand_lag_24", "demand_lag_168",
    "demand_rolling_24", "demand_rolling_168",
]
BASELINES = {
    "previous_hour": "demand_lag_1",
    "yesterday": "demand_lag_24",
    "last_week": "demand_lag_168",
}


def describe_demand_series(raw: pd.DataFrame) -> dict[str, str]:
    """Validate and describe one EIA balancing-authority or subregion series."""
    if "subba" in raw.columns:
        required = {"parent", "subba"}
        if not required.issubset(raw.columns):
            raise ValueError("Subregion demand must include parent and subba columns.")
        keys = raw[["parent", "subba"]].drop_duplicates()
        if len(keys) != 1:
            raise ValueError("Training requires exactly one EIA demand subregion.")
        parent, code = keys.iloc[0]
        name = str(raw["subba-name"].iloc[0]) if "subba-name" in raw else str(code)
        return {"kind": "subregion", "code": str(code), "name": name,
                "parent": str(parent), "label": f"{name} ({code})"}
    if "respondent" in raw.columns:
        codes = raw["respondent"].dropna().unique()
        if len(codes) != 1:
            raise ValueError("Training requires exactly one EIA balancing authority.")
        code = str(codes[0])
        name = str(raw["respondent-name"].iloc[0]) if "respondent-name" in raw else code
        return {"kind": "balancing_authority", "code": code, "name": name,
                "parent": "", "label": f"{name} ({code})"}
    raise ValueError("Input must identify its series with respondent or parent/subba columns.")


def utc_hour(value: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True)
    if pd.isna(timestamp) or timestamp != timestamp.floor("h"):
        raise ValueError("Split boundaries must be whole UTC hours.")
    return timestamp


def split_chronologically(
    frame: pd.DataFrame,
    train_start: str = "2020-01-01",
    validation_start: str = "2023-01-01",
    test_start: str = "2023-12-01",
    test_end: str = "2024-01-01",
) -> dict[str, pd.DataFrame]:
    boundaries = list(map(utc_hour, [train_start, validation_start, test_start, test_end]))
    if not all(left < right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("Require train_start < validation_start < test_start < test_end.")
    if frame["period"].duplicated().any() or not frame["period"].is_monotonic_increasing:
        raise ValueError("Feature timestamps must be unique and sorted.")
    splits = {}
    for name, start, end in zip(["train", "validation", "test"], boundaries, boundaries[1:]):
        split = frame.loc[(frame.period >= start) & (frame.period < end)].copy()
        if split.empty:
            raise ValueError(f"The {name} split has no usable examples between {start} and {end}.")
        splits[name] = split
    return splits


def regression_metrics(actual, predicted) -> dict[str, float]:
    errors = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    if not len(errors) or not np.isfinite(errors).all():
        raise ValueError("Metrics require nonempty, finite actuals and predictions.")
    return {"mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.square(errors).mean()))}


def evaluate(model, frame: pd.DataFrame, peak_threshold: float):
    predictions = frame[["period", "value"]].rename(columns={"value": "actual"}).copy()
    predictions["forecast_origin_utc"] = predictions.period - pd.Timedelta(hours=1)
    # Promote float32 predictions before CSV serialization to retain exact values.
    predictions["xgboost"] = np.asarray(model.predict(frame[FEATURE_COLUMNS]), dtype=float)
    for name, column in BASELINES.items():
        predictions[name] = frame[column]
    scores = {}
    peaks = predictions.actual >= peak_threshold
    for name in ["xgboost", *BASELINES]:
        scores[name] = regression_metrics(predictions.actual, predictions[name])
        scores[name]["peak_mae"] = (
            float((predictions.loc[peaks, name] - predictions.loc[peaks, "actual"]).abs().mean())
            if peaks.any() else None
        )
    return predictions, scores


def write_plot(predictions: pd.DataFrame, output: Path, target_label: str) -> None:
    # Avoid a GUI backend and unwritable user-level caches on headless machines.
    with tempfile.TemporaryDirectory(prefix="energy-matplotlib-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
        last_week = predictions[predictions.period >= predictions.period.max() - pd.Timedelta(days=7)]
        for column, label, color in [
            ("actual", "Actual", "#172b4d"),
            ("xgboost", "XGBoost", "#008578"),
            ("previous_hour", "Previous-hour baseline", "#b47b36"),
        ]:
            axes[0].plot(last_week.period, last_week[column], label=label, color=color, linewidth=1.3)
        axes[0].set(title=f"{target_label} demand: final seven days of held-out test", ylabel="Hourly energy (MWh)", xlabel="Target hour (UTC)")
        axes[0].legend(loc="upper left", ncol=3)
        local_hour = predictions.period.dt.tz_convert("America/Chicago").dt.hour
        for name in ["xgboost", *BASELINES]:
            hourly = (predictions[name] - predictions.actual).abs().groupby(local_hour).mean()
            axes[1].plot(hourly.index, hourly, marker=".", label=name.replace("_", " "))
        axes[1].set(title="Test MAE by Chicago hour of day", xlabel="Local hour", ylabel="MAE (MWh)", xticks=range(0, 24, 2))
        axes[1].legend(ncol=4)
        for axis in axes:
            axis.grid(alpha=0.2)
        fig.savefig(output / "test_forecast.png", dpi=160)
        plt.close(fig)


def write_report(run: dict, output: Path) -> None:
    lines = [
        "# Energy model training report", "",
        f"One-hour-ahead {run['series']['label']} demand forecast, evaluated at successive hourly origins.",
        "Past actual demand is assumed available through t-1 for every prediction at t.",
        "Weather and neighborhood weights are not model inputs. Units: hourly MWh.", "",
        "## Chronological splits", "",
        "| Split | First usable target (UTC) | Last usable target (UTC) | Usable / scheduled hours |",
        "|---|---|---|---|",
    ]
    for name, info in run["splits"].items():
        lines.append(f"| {name} | {info['first_target']} | {info['last_target']} | {info['rows']:,} / {info['scheduled_hours']:,} |")
    lines += ["", "Training uses only the training partition. Validation selects the tree count with early stopping.",
              "Test is evaluated after fitting and is never used for model selection or refitting.",
              "Missing targets and incomplete 168-hour histories are excluded; coverage is shown above.", "",
              f"Selected boosting rounds: **{run['best_iteration'] + 1}**.", "",
              "## Held-out test results", "",
              "| Model | MAE | RMSE | Peak-hour MAE |", "|---|---:|---:|---:|"]
    for name, scores in run["scores"]["test"].items():
        peak = f"{scores['peak_mae']:,.2f}" if scores["peak_mae"] is not None else "No peak hours"
        lines.append(f"| {name} | {scores['mae']:,.2f} | {scores['rmse']:,.2f} | {peak} |")
    strongest = min(BASELINES, key=lambda name: run["scores"]["test"][name]["mae"])
    baseline_mae = run["scores"]["test"][strongest]["mae"]
    model_mae = run["scores"]["test"]["xgboost"]["mae"]
    lines += ["", f"Lowest test MAE among baselines: **{strongest}**. "
              f"XGBoost {'beats' if model_mae < baseline_mae else 'does not beat'} it on this holdout.", "",
              f"Peak hours have actual demand at or above the training-only 90th percentile: {run['peak_threshold_mwh']:,.2f} MWh.", "",
              "![Test forecasts and hourly errors](test_forecast.png)", "",
              "## How to inspect the run", "",
              "- `test_predictions.csv`: actuals, forecast origins, and all four predictions for each held-out hour.",
              "- `metrics.json`: train/validation/test scores, split coverage, settings, package versions, and source checksum.",
              "- `feature_importance.csv`: model feature importance (not a causal explanation).",
              "- `model.json`: saved XGBoost model; its reload was checked against the in-memory test predictions.",
              "- `test_features.parquet`: exact model inputs for reproducing the held-out predictions.", "",
              "## Limits", "",
              "This is a rolling one-step backtest, not a month-ahead or 24-hour batch forecast. "
              "Actual prior-hour demand is supplied as each hour passes; production must account for reporting delays.",
              "One historical holdout does not establish future production accuracy. "
              "Reserve another period before further tuning based on these test results.",
              f"The target is the EIA {run['series']['kind'].replace('_', ' ')} {run['series']['label']}, not individual Chicago neighborhoods.", ""]
    (output / "report.md").write_text("\n".join(lines))


def train(args) -> dict:
    raw = pd.read_parquet(args.input)
    series = describe_demand_series(raw)
    # Always rebuild from raw data so old, compressed-timeline features cannot leak in.
    features = build_features(raw, args.max_demand, include_weather=False)
    splits = split_chronologically(features, args.train_start, args.validation_start, args.test_start, args.test_end)
    for name, frame in splits.items():
        print(f"{name:10s}: {len(frame):,} examples, {frame.period.min()} through {frame.period.max()}")

    params = dict(
        objective="reg:squarederror", eval_metric="rmse", n_estimators=args.n_estimators,
        max_depth=6, learning_rate=0.04, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        tree_method="hist", n_jobs=4, random_state=42, early_stopping_rounds=50,
    )
    model = XGBRegressor(**params)
    model.fit(
        splits["train"][FEATURE_COLUMNS], splits["train"].value,
        eval_set=[(splits["validation"][FEATURE_COLUMNS], splits["validation"].value)],
        verbose=100,
    )
    peak_threshold = float(splits["train"].value.quantile(0.9))
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output / "model.json")
    boundaries = list(map(utc_hour, [args.train_start, args.validation_start, args.test_start, args.test_end]))
    run = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "target": f"{series['label']} demand at period t (MWh)",
        "series": series, "horizon_hours": 1,
        "history_available_through": "t-1; rolling one-step evaluation",
        "feature_columns": FEATURE_COLUMNS, "calendar_timezone": "UTC",
        "max_demand": args.max_demand, "parameters": params,
        "best_iteration": int(model.best_iteration),
        "peak_threshold_mwh": peak_threshold,
        "python_version": platform.python_version(),
        "package_versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "xgboost", "scikit-learn", "pyarrow", "matplotlib"]},
        "splits": {}, "scores": {}, "saved_model_verified": False,
    }
    for (name, frame), start, end in zip(splits.items(), boundaries, boundaries[1:]):
        predictions, scores = evaluate(model, frame, peak_threshold)
        predictions.to_csv(args.output / f"{name}_predictions.csv", index=False)
        scheduled = int((end - start) / pd.Timedelta(hours=1))
        run["splits"][name] = dict(
            start_inclusive=start.isoformat(), end_exclusive=end.isoformat(),
            first_target=frame.period.min().isoformat(), last_target=frame.period.max().isoformat(),
            rows=len(frame), scheduled_hours=scheduled, coverage=len(frame) / scheduled,
        )
        run["scores"][name] = scores

    test_inputs = splits["test"][["period", *FEATURE_COLUMNS]]
    test_inputs.to_parquet(args.output / "test_features.parquet", index=False)
    restored = XGBRegressor()
    restored.load_model(args.output / "model.json")
    np.testing.assert_allclose(restored.predict(test_inputs[FEATURE_COLUMNS]), predictions.xgboost, rtol=0, atol=1e-5)
    run["saved_model_verified"] = True
    pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(args.output / "feature_importance.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
    write_plot(predictions, args.output, series["label"])
    write_report(run, args.output)
    print("\nHeld-out test (MWh; lower is better):")
    for name, scores in run["scores"]["test"].items():
        print(f"  {name:15s} MAE={scores['mae']:,.2f}  RMSE={scores['rmse']:,.2f}")
    print(f"\nSaved model reload verified. Read {args.output / 'report.md'}")
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-start", default="2020-01-01")
    parser.add_argument("--validation-start", default="2023-01-01")
    parser.add_argument("--test-start", default="2023-12-01")
    parser.add_argument("--test-end", default="2024-01-01", help="Exclusive UTC boundary")
    parser.add_argument("--max-demand", type=float, default=DEFAULT_MAX_DEMAND)
    parser.add_argument("--n-estimators", type=int, default=1200)
    args = parser.parse_args()
    if args.n_estimators <= 0:
        parser.error("--n-estimators must be positive.")
    if not args.input.is_file():
        parser.error(f"Missing raw data: {args.input}. Run ingestion/energy/fetch_eia_demand.py first.")
    try:
        train(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
