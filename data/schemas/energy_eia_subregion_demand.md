# Schema: EIA Hourly Subregion Demand

Source: EIA API v2 `electricity/rto/region-sub-ba-data` (Form EIA-930).
Fetched by: `ingestion/energy/fetch_eia_subregion_demand.py`.

The project default is `parent=PJM`, `subba=CE`: the Commonwealth Edison zone.
This is substantially closer to Chicago than PJM-wide demand, but it is still a
utility zone rather than neighborhood-level metered demand.

| Column | Type | Description |
|---|---|---|
| `period` | timezone-aware datetime | Demand hour in UTC |
| `parent` | string | Parent balancing authority; `PJM` for ComEd |
| `parent-name` | string | Parent balancing-authority name |
| `subba` | string | Subregion code; `CE` for Commonwealth Edison |
| `subba-name` | string | Subregion name |
| `value` | float | Hourly demand in megawatthours |
| `value-units` | string | Expected to be `megawatthours` |

Values are preliminary operational data. Missing or invalid observations remain
missing during feature engineering; the pipeline never interpolates targets.
EIA's `end` parameter is inclusive of the requested date's hourly observations.
