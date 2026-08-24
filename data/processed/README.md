# data/processed/

Cleaned, joined, feature-engineered data ready to feed into model training. This is the output of scripts in `features/`, not something edited by hand.

Suggested layout as this grows, mirroring `features/`:
```
processed/
├── transit/
├── energy/
└── weather/
```

Same rule as `data/raw/`: don't commit large files to git, use R2/BigQuery for anything meant to be shared.
