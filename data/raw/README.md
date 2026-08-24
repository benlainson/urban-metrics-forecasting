# data/raw/

Unmodified API responses and downloaded files, exactly as pulled from source. Never edit these by hand — if a fetch script's output was wrong, fix the script and re-run it.

Suggested layout as this grows:
```
raw/
├── transit/    # CTA bus/train/ridership pulls
├── energy/     # EIA + utility API pulls
└── weather/    # Open-Meteo pulls
```

Large files (parquet, csv) should not be committed — add patterns to the root `.gitignore` and rely on Cloudflare R2 / BigQuery as the shared source of truth.
