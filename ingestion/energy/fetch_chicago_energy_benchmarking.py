"""
ingestion/energy/fetch_chicago_energy_benchmarking.py

Pulls the Chicago Energy Benchmarking dataset (annual building-level
energy use, reported under the city's benchmarking ordinance) from the
Chicago Data Portal (Socrata) and writes it to data/raw/energy/.

This dataset is annual and per-building, NOT hourly. It's meant to be
joined against EIA hourly demand (see fetch_eia_demand.py) to give the
grid-load model neighborhood/community-area weighting — not used as a
time series on its own.

Dataset: https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking/xq83-jr8c

Setup:
    Works without a token but is rate-limited. For higher limits, register
    an app token at https://data.cityofchicago.org/profile/app_tokens and
    set CHICAGO_APP_TOKEN in your environment / .env.

Usage:
    python ingestion/energy/fetch_chicago_energy_benchmarking.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import requests

DATASET_ID = "xq83-jr8c"
BASE_URL = f"https://data.cityofchicago.org/resource/{DATASET_ID}.json"
PAGE_SIZE = 50000
OUTPUT_DIR = Path("data/raw/energy")


def fetch_all(app_token: str | None) -> pd.DataFrame:
    headers = {"X-App-Token": app_token} if app_token else {}
    all_rows = []
    offset = 0

    while True:
        params = {"$limit": PAGE_SIZE, "$offset": offset, "$order": "data_year DESC"}
        resp = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        rows = resp.json()

        if not rows:
            break

        all_rows.extend(rows)
        print(f"  fetched {len(rows)} rows (offset={offset}, total so far={len(all_rows)})")

        if len(rows) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    return pd.DataFrame(all_rows)


def main():
    app_token = os.environ.get("CHICAGO_APP_TOKEN")
    if not app_token:
        print("No CHICAGO_APP_TOKEN set — proceeding without one (lower rate limit).")

    print("Fetching Chicago Energy Benchmarking dataset...")
    df = fetch_all(app_token)

    if df.empty:
        print("No data returned.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "chicago_energy_benchmarking.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\nSaved {df.shape[0]} rows x {df.shape[1]} cols to {out_path}")
    print("\nColumns:", list(df.columns))
    if "data_year" in df.columns:
        print("\nYear range:", df["data_year"].min(), "to", df["data_year"].max())
    print("\nHead:")
    print(df.head())


if __name__ == "__main__":
    main()
