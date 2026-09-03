# Schema: Chicago Energy Benchmarking (`chicago_energy_benchmarking.parquet`)

Source: Chicago Data Portal (Socrata), dataset `xq83-jr8c`.
Fetched by: `ingestion/energy/fetch_chicago_energy_benchmarking.py`

Key columns (full set varies slightly by year — Socrata returns
whatever fields are populated, check `df.columns` after a fresh pull):

| Column | Type | Description |
|---|---|---|
| `data_year` | int (as string from API) | Year the energy use applies to, not the submission/download year |
| `id` | string | Building ID |
| `property_name` | string | Building name |
| `address` | string | Street address |
| `zip_code` | string | ZIP code |
| `community_area` | string | Chicago community area name; trim and normalize case before aggregation |
| `primary_property_type` | string | e.g. Office, Multifamily Housing |
| `gross_floor_area_buildings_sq_ft` | numeric string | Building floor area (sq ft) |
| `site_eui_kbtu_sq_ft` | numeric string | Site energy use intensity, all fuels (kBtu/sq ft) |
| `source_eui_kbtu_sq_ft` | numeric string | Source energy use intensity (kBtu/sq ft) |
| `electricity_use_kbtu` | numeric string | Annual electricity use (kBtu, not kWh) |
| `natural_gas_use_kbtu` | numeric string | Annual natural gas use (kBtu) |
| `ghg_intensity_kg_co2e_sq_ft` | numeric string | Greenhouse gas intensity (kg CO2e/sq ft) |

**Granularity:** one row per building per reporting year — **annual**,
not hourly. Covers buildings >50,000 sqft (less than 1% of buildings,
~20% of total building energy use), so it's a partial but weighted
sample, not a census.

**Analysis:** select one data year before comparing annual community totals;
do not add all reporting years and label the result as the latest year.
Convert numeric strings explicitly and keep missing reports missing. Check
property/year uniqueness before aggregation. For electricity per floor area,
divide summed electricity by summed floor area for the same eligible properties.
This is distinct from all-fuel site EUI.

**Model boundary:** this dataset is not an input to the current XGBoost model.
The experimental weighting helper in `features/energy/` aggregates across years;
its output is not a validated local demand estimate. Allocating the entire PJM
regional total among Chicago communities would be geographically incorrect.
Local forecasting requires an appropriate local total and independent validation.

The notebook determines the latest year from the file. A refresh on September 2,
2026 returned data through 2023; do not relabel those records as 2026 usage.
See the [official dataset description](https://catalog.data.gov/dataset/chicago-energy-benchmarking).
