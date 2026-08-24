# Schema: Chicago Energy Benchmarking (`chicago_energy_benchmarking.parquet`)

Source: Chicago Data Portal (Socrata), dataset `xq83-jr8c`.
Fetched by: `ingestion/energy/fetch_chicago_energy_benchmarking.py`

Key columns (full set varies slightly by year — Socrata returns
whatever fields are populated, check `df.columns` after a fresh pull):

| Column | Type | Description |
|---|---|---|
| `data_year` | int (as string from API) | Reporting year |
| `id` | string | Building ID |
| `property_name` | string | Building name |
| `address` | string | Street address |
| `zip_code` | string | ZIP code |
| `community_area` | string | Chicago community area name — **this is the join key for neighborhood-level features** |
| `primary_property_type` | string | e.g. Office, Multifamily Housing |
| `gross_floor_area` | float | Square footage |
| `site_eui` | float | Site Energy Use Intensity (kBtu/sqft) |
| `source_eui` | float | Source Energy Use Intensity (kBtu/sqft) |
| `electricity_use` | float | kWh |
| `natural_gas_use` | float | kBtu |
| `ghg_intensity` | float | Greenhouse gas intensity |

**Granularity:** one row per building per reporting year — **annual**,
not a time series. Covers buildings >50,000 sqft (~1% of buildings,
~20% of total building energy use), so it's a partial but weighted
sample, not a census.

**Use in `features/energy/`:** aggregate `electricity_use` /
`site_eui` by `community_area` to build a static (or slowly-updating)
neighborhood weighting, then apply that weighting to the EIA hourly
demand series rather than trying to use this as its own time series.
