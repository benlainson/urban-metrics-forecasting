"""Fetch hourly EIA demand for one balancing-authority subregion.

The default is the Commonwealth Edison (CE) zone inside PJM, a closer match
for Chicago than the full PJM balancing authority. Run from the repository root:

    python ingestion/energy/fetch_eia_subregion_demand.py \
        --parent PJM --subregion CE --start 2020-01-01 --end 2026-01-01
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


EIA_BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/"
PAGE_SIZE = 5000
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data/raw/energy"


def fetch_page(
    api_key: str,
    parent: str,
    subregion: str,
    offset: int,
    start: str | None,
    end: str | None,
) -> dict:
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[parent][]": parent,
        "facets[subba][]": subregion,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": offset,
        "length": PAGE_SIZE,
    }
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    response = requests.get(EIA_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def normalize_rows(rows: list[dict], parent: str, subregion: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    required = {"period", "parent", "subba", "value", "value-units"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"EIA response is missing required columns: {sorted(missing)}")
    frame["period"] = pd.to_datetime(frame["period"], utc=True, errors="raise")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if not frame["parent"].eq(parent).all() or not frame["subba"].eq(subregion).all():
        raise ValueError("EIA returned a different parent/subregion than requested.")
    if frame["period"].duplicated().any():
        raise ValueError("EIA returned duplicate subregion timestamps.")
    if not frame["period"].is_monotonic_increasing:
        raise ValueError("EIA subregion timestamps are not sorted.")
    if not frame["value-units"].eq("megawatthours").all():
        raise ValueError("EIA returned unexpected demand units.")
    return frame


def fetch_all(
    api_key: str,
    parent: str,
    subregion: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    rows: list[dict] = []
    offset = 0
    while True:
        payload = fetch_page(api_key, parent, subregion, offset, start, end)
        page = payload.get("response", {}).get("data", [])
        if not page:
            break
        rows.extend(page)
        print(f"  fetched {len(page)} rows (offset={offset}, total so far={len(rows)})")
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.2)
    return normalize_rows(rows, parent, subregion)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch hourly EIA subregion demand.")
    parser.add_argument("--parent", default="PJM", help="Parent balancing authority (default: PJM)")
    parser.add_argument("--subregion", default="CE", help="Subregion code (default: CE, Commonwealth Edison)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=False)
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        parser.error("EIA_API_KEY not found in the environment or repository .env file.")
    parent = args.parent.strip().upper()
    subregion = args.subregion.strip().upper()
    output = args.output or OUTPUT_DIR / (
        "eia_demand_comed.parquet" if (parent, subregion) == ("PJM", "CE")
        else f"eia_demand_{parent.lower()}_{subregion.lower()}.parquet"
    )
    print(f"Fetching hourly demand for parent={parent} subregion={subregion} "
          f"start={args.start} end={args.end} ...")
    try:
        frame = fetch_all(api_key, parent, subregion, args.start, args.end)
    except requests.RequestException as exc:
        # Request exception strings may contain the request URL and API key.
        print(f"ERROR: EIA request failed ({type(exc).__name__}); request details suppressed.", file=sys.stderr)
        raise SystemExit(1) from exc
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if frame.empty:
        parser.error("No data returned. Check the codes, dates, and API key.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(output)
    print(f"\nSaved {len(frame):,} rows x {len(frame.columns)} cols to {output}")
    print(f"Date range: {frame.period.min()} to {frame.period.max()}")
    print(f"Series: {frame['subba-name'].iloc[0] if 'subba-name' in frame else subregion}")


if __name__ == "__main__":
    main()
