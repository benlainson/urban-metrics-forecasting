"""
features/energy/engineer.py

Builds model-ready features from raw hourly PJM grid demand
(data/raw/energy/eia_demand_pjm.parquet), following the same pattern as
the transit feature engineering step in the project plan: calendar
features, lag features, rolling averages.

Also folds in:
  - Outlier cleaning (see notebooks/02_eda_energy.ipynb for why this is
    needed — the raw EIA pull can contain a small number of corrupted
    readings that are orders of magnitude too large).
  - An optional join against weather features, per CLAUDE.md's
    cross-domain dependency: features/weather/ is a producer for this
    domain. If the weather teammate hasn't published
    data/processed/weather/features.parquet yet, this script skips the
    join and prints a warning rather than failing, so energy work isn't
    blocked on weather being done first.
  - An optional join against the Chicago Energy Benchmarking dataset for
    a static per-community-area weighting (see
    data/schemas/energy_chicago_benchmarking.md for why this is a
    weighting, not a time series, feature).

Usage:
    python features/energy/engineer.py
    python features/energy/engineer.py --max-demand 300000
"""

import argparse
from pathlib import Path

import pandas as pd

RAW_DEMAND_PATH = Path("data/raw/energy/eia_demand_pjm.parquet")
RAW_BENCHMARKING_PATH = Path("data/raw/energy/chicago_energy_benchmarking.parquet")
WEATHER_FEATURES_PATH = Path("data/processed/weather/features.parquet")
OUTPUT_PATH = Path("data/processed/energy/features.parquet")

# PJM's actual historical max hourly demand is roughly 165,000 MWh.
# Anything wildly beyond that is a data error, not a real demand spike.
# See notebooks/02_eda_energy.ipynb section 3-4 for how this was found.
DEFAULT_MAX_DEMAND = 300_000


def clean_outliers(df: pd.DataFrame, max_demand: float) -> pd.DataFrame:
    before = len(df)
    df = df[df["value"] < max_demand].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} outlier rows (value >= {max_demand:,})")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["period"].dt.hour
    df["dayofweek"] = df["period"].dt.dayofweek
    df["month"] = df["period"].dt.month
    df["year"] = df["period"].dt.year
    df["quarter"] = df["period"].dt.quarter
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["week_of_year"] = df["period"].dt.isocalendar().week.astype(int)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("period").reset_index(drop=True)

    # Lag features — what demand was 1 hour ago, 1 day ago (same hour), 1 week ago (same hour+day)
    df["demand_lag_1"] = df["value"].shift(1)
    df["demand_lag_24"] = df["value"].shift(24)
    df["demand_lag_168"] = df["value"].shift(168)

    # Rolling averages — smooths noise, captures short-term trend
    df["demand_rolling_24"] = df["value"].shift(1).rolling(24).mean()
    df["demand_rolling_168"] = df["value"].shift(1).rolling(168).mean()

    return df


def load_weather_features() -> pd.DataFrame | None:
    if not WEATHER_FEATURES_PATH.exists():
        print(f"  No weather features found at {WEATHER_FEATURES_PATH} — "
              "skipping weather join. Re-run this script once "
              "features/weather/ has published its output.")
        return None
    weather = pd.read_parquet(WEATHER_FEATURES_PATH)
    print(f"  Loaded weather features: {weather.shape[0]} rows, columns: {list(weather.columns)}")
    return weather


def join_weather(df: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    if weather is None:
        return df
    if "period" not in weather.columns:
        print("  WARNING: weather features have no 'period' column to join on — skipping join. "
              "Coordinate with the weather teammate on a shared join key.")
        return df
    before_cols = set(df.columns)
    df = df.merge(weather, on="period", how="left")
    new_cols = set(df.columns) - before_cols
    print(f"  Joined weather features, added columns: {sorted(new_cols)}")
    return df


def load_community_area_weighting() -> pd.DataFrame | None:
    """Static per-community-area electricity-use weighting from the
    (annual, building-level) Chicago Energy Benchmarking dataset.

    NOTE: this is intentionally NOT joined onto the hourly demand series
    directly — there's no per-hour, per-neighborhood key to join on yet.
    It's returned separately so it can be used downstream (in
    models/energy/) to disaggregate a BA-level forecast into a
    neighborhood-level estimate. See data/schemas/energy_chicago_benchmarking.md.
    """
    if not RAW_BENCHMARKING_PATH.exists():
        print(f"  No benchmarking data found at {RAW_BENCHMARKING_PATH} — "
              "run ingestion/energy/fetch_chicago_energy_benchmarking.py first.")
        return None

    bench = pd.read_parquet(RAW_BENCHMARKING_PATH)
    bench["community_area"] = bench["community_area"].str.strip().str.upper()
    bench["electricity_use_kbtu"] = pd.to_numeric(bench["electricity_use_kbtu"], errors="coerce")

    weighting = (
        bench.groupby("community_area")["electricity_use_kbtu"]
        .sum()
        .reset_index()
    )
    total = weighting["electricity_use_kbtu"].sum()
    weighting["community_area_weight"] = weighting["electricity_use_kbtu"] / total
    return weighting.sort_values("community_area_weight", ascending=False)


def build_features(df: pd.DataFrame, max_demand: float) -> pd.DataFrame:
    print("Cleaning outliers...")
    df = clean_outliers(df, max_demand)

    print("Adding calendar features...")
    df = add_calendar_features(df)

    print("Adding lag / rolling features...")
    df = add_lag_and_rolling_features(df)

    print("Loading weather features (if available)...")
    weather = load_weather_features()
    df = join_weather(df, weather)

    # Drop rows with NaNs introduced by lag/rolling features (first ~168 hours)
    # and by outlier removal creating small gaps.
    before = len(df)
    df = df.dropna(subset=["demand_lag_1", "demand_lag_24", "demand_lag_168",
                            "demand_rolling_24", "demand_rolling_168"])
    print(f"  Dropped {before - len(df)} rows with NaN lag/rolling values (expected — warm-up period)")

    return df


def main():
    parser = argparse.ArgumentParser(description="Build energy demand features.")
    parser.add_argument("--max-demand", type=float, default=DEFAULT_MAX_DEMAND,
                         help=f"Upper bound for valid hourly demand (default: {DEFAULT_MAX_DEMAND:,})")
    args = parser.parse_args()

    if not RAW_DEMAND_PATH.exists():
        print(f"ERROR: {RAW_DEMAND_PATH} not found. "
              f"Run ingestion/energy/fetch_eia_demand.py first.")
        return

    print(f"Loading raw demand data from {RAW_DEMAND_PATH}...")
    df = pd.read_parquet(RAW_DEMAND_PATH)
    print(f"  {df.shape[0]} rows loaded")

    features = build_features(df, args.max_demand)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {features.shape[0]} rows x {features.shape[1]} cols to {OUTPUT_PATH}")
    print("\nColumns:", list(features.columns))
    print("\nHead:")
    print(features.head())

    # Also compute and save the (separate) community-area weighting table
    weighting = load_community_area_weighting()
    if weighting is not None:
        weighting_path = Path("data/processed/energy/community_area_weighting.parquet")
        weighting.to_parquet(weighting_path, index=False)
        print(f"\nSaved community-area weighting to {weighting_path}")
        print(weighting.head(10))


if __name__ == "__main__":
    main()
