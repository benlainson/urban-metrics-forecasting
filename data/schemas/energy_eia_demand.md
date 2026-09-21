# Schema: EIA Hourly Grid Demand (`eia_demand_<respondent>.parquet`)

Source: EIA API v2, `electricity/rto/region-data` route.
Fetched by: `ingestion/energy/fetch_eia_demand.py`

| Column | Type | Description |
|---|---|---|
| `period` | datetime (hourly, UTC) | Hour the demand value applies to |
| `respondent` | string | Balancing authority code (default `PJM`, covers ComEd/Chicago) |
| `respondent-name` | string | Human-readable BA name |
| `type` | string | Always `D` (Demand) for this pull |
| `type-name` | string | Human-readable type label |
| `value` | float | Demand in megawatthours |
| `value-units` | string | Should always be `megawatthours` |

**Known limitation:** this is balancing-authority-level demand, not
neighborhood-level. PJM does not publish a Chicago-neighborhood
subdivision of hourly demand. See `energy_chicago_benchmarking.md` for
the complementary annual/building-level dataset used to approximate
spatial granularity.

**Nullability:** `value` can be null for a small number of hours. Feature
engineering restores the complete UTC timeline and leaves missing/invalid demand
as null. It does not interpolate targets; examples with incomplete histories are
excluded after lag and rolling features are calculated.
