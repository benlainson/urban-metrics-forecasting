"""
ingestion/energy/fetch_eia_demand.py

Pulls hourly electricity demand for the PJM balancing authority (the RTO
that covers ComEd / the Chicago area) from the EIA API v2, and writes it
to data/raw/energy/ as parquet.

This is balancing-authority-level demand, NOT neighborhood-level. See
data/schemas/energy_eia_demand.md for why, and features/energy/README.md
for how this gets combined with the (annual, building-level) Chicago
Energy Benchmarking dataset to approximate neighborhood granularity.

Setup:
    1. Register for a free API key: https://www.eia.gov/opendata/
    2. Put it in a local .env file (gitignored) as EIA_API_KEY=...
       or export it: export EIA_API_KEY=your_key_here
    3. In CI, set it as a GitHub Actions secret (see .github/workflows/).

Usage:
    python ingestion/energy/fetch_eia_demand.py
    python ingestion/energy/fetch_eia_demand.py --start 2020-01-01 --end 2024-01-01
    python ingestion/energy/fetch_eia_demand.py --respondent PJM
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

EIA_BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
DEFAULT_RESPONDENT = "PJM"   # Balancing authority covering ComEd / Chicago
PAGE_SIZE = 5000             # EIA API v2 max rows per request
OUTPUT_DIR = Path("data/raw/energy")


def fetch_page(api_key: str, respondent: str, offset: int, start: str | None, end: str | None) -> dict:
    """Fetch a single page of hourly demand data from the EIA API."""
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "facets[type][]": "D",  # D = Demand (as opposed to generation, interchange, etc.)
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": offset,
        "length": PAGE_SIZE,
    }
    if start:
        params["start"] = start
    if end:
        params["end"] = end

    resp = requests.get(EIA_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_all(api_key: str, respondent: str, start: str | None, end: str | None) -> pd.DataFrame:
    """Page through the EIA API until exhausted and return one DataFrame."""
    all_rows = []
    offset = 0

    while True:
        payload = fetch_page(api_key, respondent, offset, start, end)
        rows = payload.get("response", {}).get("data", [])
        if not rows:
            break

        all_rows.extend(rows)
        print(f"  fetched {len(rows)} rows (offset={offset}, total so far={len(all_rows)})")

        if len(rows) < PAGE_SIZE:
            break  # last page

        offset += PAGE_SIZE
        time.sleep(0.2)  # be polite to the API

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # EIA returns 'value' as a string (see API changelog v2.1.6) — cast it.
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"])
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch hourly EIA grid demand data.")
    parser.add_argument("--respondent", default=DEFAULT_RESPONDENT,
                         help="Balancing authority code (default: PJM, covers Chicago/ComEd)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (optional)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (optional)")
    args = parser.parse_args()

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print("ERROR: EIA_API_KEY environment variable not set.")
        print("Register at https://www.eia.gov/opendata/ and export EIA_API_KEY=your_key")
        sys.exit(1)

    print(f"Fetching hourly demand for respondent={args.respondent} "
          f"start={args.start} end={args.end} ...")

    df = fetch_all(api_key, args.respondent, args.start, args.end)

    if df.empty:
        print("No data returned. Check respondent code, date range, and API key.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"eia_demand_{args.respondent.lower()}.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\nSaved {df.shape[0]} rows x {df.shape[1]} cols to {out_path}")
    print("\nColumns:", list(df.columns))
    print("\nDate range:", df["period"].min(), "to", df["period"].max())
    print("\nHead:")
    print(df.head())


if __name__ == "__main__":
    main()
