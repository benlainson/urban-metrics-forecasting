# 🏙️ Urban Pulse — Project Plan
> Predictive demand forecasting for urban infrastructure

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Phase 0 — Environment Setup](#phase-0--environment-setup-day-1)
- [Phase 1 — Data Foundation](#phase-1--data-foundation-days-210)
- [Phase 2 — Baseline Models](#phase-2--baseline-models-days-1120)
- [Phase 3 — Advanced Modeling](#phase-3--advanced-modeling-days-2135)
- [Phase 4 — Serving Layer](#phase-4--serving-layer-days-3645)
- [Phase 5 — Frontend](#phase-5--frontend-days-4655)
- [Phase 6 — Resume Polish](#phase-6--resume-polish-days-5660)
- [Timeline Summary](#timeline-summary)
- [Tech Stack Decisions](#tech-stack-decisions)

---

## Project Overview

Urban Pulse is a multi-modal forecasting platform that ingests real-time and historical city data, runs predictive ML models, and serves forecasts through a public dashboard and REST API.

**What it predicts:**

| Domain | Prediction Target | Data Source |
|---|---|---|
| 🚇 Transit | Ridership demand per station per hour | Chicago CTA open APIs |
| ⚡ Energy | Grid load by neighborhood | EIA + utility APIs |
| 🏥 ER Wait Times | Wait time 2–4 hrs ahead | State health portals |
| 🅿️ Parking | Occupancy by zone | City open data portals |
| 🚦 Traffic | Congestion hotspots | Open-Meteo + city APIs |

**Starting focus:** Transit ridership + energy load. These have the richest open datasets and most modeling complexity.

---

## Architecture

```
[City APIs / Open Data]
        ↓
[Ingestion Layer - GitHub Actions Cron]
        ↓
[Raw Storage - Cloudflare R2]
        ↓
[Analytics DB - BigQuery]
        ↓
[Feature Engineering - Pandas / Parquet]
        ↓
[Feature Cache - Upstash Redis]
        ↓
[Model Training - XGBoost / TFT on Kaggle + GCP]
        ↓
[Experiment Tracking - Weights & Biases]
        ↓
[Model Registry - W&B Artifacts]
        ↓
[Serving API - FastAPI + Docker on Cloud Run]
        ↓
[Frontend Dashboard - Next.js + Deck.gl on Vercel]
```

---

## Phase 0 — Environment Setup (Day 1)

> Do not skip this. A clean setup saves hours of pain later.

### Step 1 — Install Core Tools

```bash
# Install in this order
Python 3.11+      →  python.org
Node.js 20 LTS    →  nodejs.org
Docker Desktop    →  docker.com
VS Code           →  code.visualstudio.com
Git               →  git-scm.com
```

### Step 2 — VS Code Extensions

```
Python (Microsoft)
Pylance
Docker
GitLens
Jupyter
Thunder Client       # API testing, replaces Postman
```

### Step 3 — Create Accounts (All Free)

```
GitHub             →  github.com
Google Cloud       →  cloud.google.com        ← grab $300 free credit on signup
Cloudflare         →  cloudflare.com
Weights & Biases   →  wandb.ai
Kaggle             →  kaggle.com
Vercel             →  vercel.com
Hugging Face       →  huggingface.co
Clerk              →  clerk.com
Upstash            →  upstash.com
```

### Step 4 — Create Project Repo

```bash
mkdir urban-pulse
cd urban-pulse
git init
```

Set up this folder structure immediately:

```
urban-pulse/
├── .github/
│   └── workflows/          # GitHub Actions pipelines
├── data/
│   ├── raw/                # Raw API responses
│   ├── processed/          # Cleaned features
│   └── schemas/            # Data contracts / column definitions
├── ingestion/              # API polling scripts
├── features/               # Feature engineering
├── models/                 # Training scripts
├── serving/                # FastAPI application
├── frontend/               # Next.js app
├── notebooks/              # EDA and experimentation
└── docs/                   # Architecture notes and ADRs
```

---

## Phase 1 — Data Foundation (Days 2–10)

### Step 5 — Pick Your City and Register for APIs

**Start with Chicago.** Best open data portal, most reliable APIs, longest historical records.

```
Register →  data.cityofchicago.org
Create an app token (free — removes rate limits)

APIs to pull:
1. CTA Bus Tracker API         →  real-time bus positions
2. CTA Train Tracker API       →  real-time train arrivals
3. Chicago Transit Ridership   →  historical dataset (2001–present)
4. Chicago Energy Benchmarking →  building energy use by block
```

### Step 6 — Pull Your First Dataset Manually

```python
# ingestion/fetch_cta_ridership.py
import requests
import pandas as pd

APP_TOKEN = "your_token_here"

url = "https://data.cityofchicago.org/resource/t2rn-p8d7.json"
params = {
    "$limit": 50000,
    "$order": "service_date DESC",
    "$$app_token": APP_TOKEN
}

response = requests.get(url, params=params)
df = pd.DataFrame(response.json())
df.to_parquet("data/raw/cta_ridership.parquet")

print(df.head())
print(df.dtypes)
print(df.shape)
```

> Run this. Look at what comes back. Understand every column before moving on.

### Step 7 — Set Up BigQuery

```bash
# Install Google Cloud CLI
→  cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create your dataset
gcloud bigquery datasets create urban_pulse \
  --location=US
```

```python
# ingestion/push_to_bigquery.py
from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

df = pd.read_parquet("data/raw/cta_ridership.parquet")
table_id = "your_project.urban_pulse.cta_ridership_raw"

job = client.load_table_from_dataframe(df, table_id)
job.result()
print(f"Loaded {df.shape[0]} rows into {table_id}")
```

### Step 8 — Automate Ingestion with GitHub Actions

```yaml
# .github/workflows/ingest_cta.yml
name: CTA Daily Ingestion

on:
  schedule:
    - cron: '0 6 * * *'    # Runs 6am UTC daily
  workflow_dispatch:         # Allows manual trigger

jobs:
  ingest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run ingestion
        env:
          CTA_APP_TOKEN: ${{ secrets.CTA_APP_TOKEN }}
          GOOGLE_CREDENTIALS: ${{ secrets.GOOGLE_CREDENTIALS }}
        run: python ingestion/fetch_cta_ridership.py
```

> Add secrets in GitHub → Settings → Secrets. Your pipeline now runs automatically every day without touching a server.

### Step 9 — Exploratory Data Analysis

```python
# notebooks/01_eda_ridership.ipynb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("data/raw/cta_ridership.parquet")

# Questions to answer in this notebook:

# 1. What is the date range?
print(df['service_date'].min(), df['service_date'].max())

# 2. Which stations have highest ridership?
df.groupby('stationname')['rides'].sum().sort_values(ascending=False).head(10)

# 3. Is there weekly seasonality?
df['dayofweek'] = pd.to_datetime(df['service_date']).dt.dayofweek
df.groupby('dayofweek')['rides'].mean().plot()

# 4. Is there yearly seasonality?
df['month'] = pd.to_datetime(df['service_date']).dt.month
df.groupby('month')['rides'].mean().plot()

# 5. Are there anomalies? (COVID drop, holidays, events)
df.set_index('service_date')['rides'].plot(figsize=(20, 5))
```

> **Do not move to Phase 2 until you can answer:** What does ridership look like by station, by day, by month, and by year? Where are the anomalies?

---

## Phase 2 — Baseline Models (Days 11–20)

### Step 10 — Feature Engineering

```python
# features/engineer.py
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['service_date'])

    # Time features
    df['dayofweek']    = df['date'].dt.dayofweek
    df['month']        = df['date'].dt.month
    df['year']         = df['date'].dt.year
    df['quarter']      = df['date'].dt.quarter
    df['is_weekend']   = df['dayofweek'].isin([5, 6]).astype(int)
    df['week_of_year'] = df['date'].dt.isocalendar().week

    # Lag features — what happened yesterday, last week
    df = df.sort_values('date')
    df['rides_lag_1']  = df.groupby('station_id')['rides'].shift(1)
    df['rides_lag_7']  = df.groupby('station_id')['rides'].shift(7)
    df['rides_lag_14'] = df.groupby('station_id')['rides'].shift(14)

    # Rolling averages
    df['rides_rolling_7']  = df.groupby('station_id')['rides'] \
                               .transform(lambda x: x.rolling(7).mean())
    df['rides_rolling_30'] = df.groupby('station_id')['rides'] \
                               .transform(lambda x: x.rolling(30).mean())

    return df.dropna()
```

### Step 11 — Train Prophet Baseline

```python
# models/train_prophet.py
from prophet import Prophet
import pandas as pd
import wandb

wandb.init(project="urban-pulse", name="prophet-baseline")

df = pd.read_parquet("data/processed/features.parquet")

# Prophet requires specific column names
station_df = df[df['stationname'] == 'Lake/State'][['date', 'rides']]
station_df.columns = ['ds', 'y']

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

train = station_df[station_df['ds'] < '2023-01-01']
test  = station_df[station_df['ds'] >= '2023-01-01']

model.fit(train)

future   = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test['y'], forecast.tail(len(test))['yhat'])

wandb.log({"mae": mae, "model": "prophet"})
print(f"Prophet MAE: {mae:.2f}")
```

### Step 12 — Train XGBoost and Beat Prophet

```python
# models/train_xgboost.py
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import wandb
import pandas as pd

wandb.init(project="urban-pulse", name="xgboost-v1")

df = pd.read_parquet("data/processed/features.parquet")

features = [
    'dayofweek', 'month', 'year', 'is_weekend',
    'rides_lag_1', 'rides_lag_7', 'rides_lag_14',
    'rides_rolling_7', 'rides_rolling_30'
]

train = df[df['date'] < '2023-01-01']
test  = df[df['date'] >= '2023-01-01']

model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8
)

model.fit(
    train[features], train['rides'],
    eval_set=[(test[features], test['rides'])],
    early_stopping_rounds=50,
    verbose=100
)

preds = model.predict(test[features])
mae   = mean_absolute_error(test['rides'], preds)

wandb.log({"mae": mae, "model": "xgboost"})
print(f"XGBoost MAE: {mae:.2f}")
model.save_model("models/xgboost_v1.json")
```

> **Goal:** XGBoost should beat Prophet. If it doesn't, go back and engineer better lag features.

---

## Phase 3 — Advanced Modeling (Days 21–35)

### Step 13 — Add Weather Data as External Features

```python
# ingestion/fetch_weather.py
# Open-Meteo is completely free — no API key required
import openmeteo_requests

url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 41.85,      # Chicago
    "longitude": -87.65,
    "daily": [
        "temperature_2m_max",
        "precipitation_sum",
        "snowfall_sum"
    ],
    "timezone": "America/Chicago",
    "start_date": "2020-01-01",
    "end_date": "2024-01-01"
}
# Merge this into your feature set
# Hypothesis: ridership drops significantly when it snows
```

### Step 14 — Add Anomaly Detection

```python
# models/anomaly.py
from sklearn.ensemble import IsolationForest
import pandas as pd

df = pd.read_parquet("data/processed/features.parquet")

model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_score'] = model.fit_predict(df[['rides', 'rides_rolling_7']])
df['is_anomaly']    = df['anomaly_score'] == -1

# These are your most interesting data points
# COVID lockdowns, blizzards, and major events will surface here
print(df[df['is_anomaly']].sort_values('date'))
```

### Step 15 — Train Temporal Fusion Transformer

> Run this in a Kaggle GPU notebook — paste the code there for free GPU access.

```python
# Run on Kaggle GPU notebook
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE
import pytorch_lightning as pl

# Key hyperparameters to log in W&B:
# - hidden_size:        64
# - attention_head_size: 4
# - dropout:            0.1
# - learning_rate:      0.001

# Full implementation reference:
# https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html
```

---

## Phase 4 — Serving Layer (Days 36–45)

### Step 16 — Build the FastAPI Server

```python
# serving/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd

app = FastAPI(title="Urban Pulse API", version="1.0")

model = xgb.XGBRegressor()
model.load_model("models/xgboost_v1.json")

class PredictionRequest(BaseModel):
    station_id: str
    date: str

class PredictionResponse(BaseModel):
    station_id: str
    date: str
    predicted_rides: float
    confidence_lower: float
    confidence_upper: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    # Build features from request
    # Run model inference
    # Return prediction with confidence interval
    pass

@app.get("/anomalies")
def get_anomalies(start_date: str, end_date: str):
    # Return detected anomalies in date range
    pass
```

### Step 17 — Dockerize

```dockerfile
# serving/Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

```bash
# Test locally first
docker build -t urban-pulse-api .
docker run -p 8080:8080 urban-pulse-api

# Hit it with Thunder Client in VS Code
GET http://localhost:8080/health
```

### Step 18 — Deploy to Cloud Run

```bash
gcloud run deploy urban-pulse-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --max-instances 3
```

---

## Phase 5 — Frontend (Days 46–55)

### Step 19 — Scaffold Next.js App

```bash
cd frontend
npx create-next-app@latest . --typescript --tailwind --app
npm install @deck.gl/react maplibre-gl recharts
```

### Step 20 — Build These Views in Order

```
1. /                  →  Landing page, project summary
2. /dashboard         →  Main map view with live predictions
3. /stations/[id]     →  Per-station forecast chart
4. /anomalies         →  Anomaly timeline view
5. /api-docs          →  Link to FastAPI swagger docs (/docs)
```

### Step 21 — Deploy Frontend to Vercel

```bash
npm install -g vercel
vercel --prod
# Live in 90 seconds
```

---

## Phase 6 — Resume Polish (Days 56–60)

### Step 22 — Write Your README

```markdown
# Urban Pulse 🏙️
> Predictive demand forecasting for urban infrastructure

## What it does
## Architecture diagram
## Models and accuracy comparison
## Live demo link
## How to run locally
## Dataset sources
```

### Step 23 — Write One Technical Blog Post

Post on **dev.to** or **Substack** (both free):

```
Title: "How I built a city-scale ML forecasting platform for $0/month"

Cover:
- The problem
- Data pipeline architecture
- Model progression (Prophet → XGBoost → TFT)
- Accuracy improvements at each stage
- Lessons learned
```

### Step 24 — Final Checklist Before Adding to Resume

```
✅ Live demo URL works
✅ GitHub repo is public with clean README
✅ W&B project is public (shows full experiment history)
✅ Blog post is live
✅ FastAPI swagger docs are live (/docs endpoint)
✅ At least 3 models trained and compared
✅ Anomaly detection working
✅ Real city data — not synthetic
```

---

## Timeline Summary

| Phase | Days | Deliverable |
|---|---|---|
| 0 — Setup | 1 | Clean environment, all accounts created |
| 1 — Data | 2–10 | Pipeline running, data in BigQuery, EDA complete |
| 2 — Baseline | 11–20 | Prophet + XGBoost trained and logged in W&B |
| 3 — Advanced | 21–35 | TFT trained, anomaly detection live |
| 4 — Serving | 36–45 | FastAPI live on Cloud Run |
| 5 — Frontend | 46–55 | Next.js dashboard live on Vercel |
| 6 — Polish | 56–60 | README, blog post, resume-ready |

**60 days. Real data. Real models. Real deployment. ~$0/month.**

---

## Tech Stack Decisions

Every tool was chosen against two criteria: **highest quality** and **lowest cost**. Below is the full decision log including every alternative considered.

---

### Data Ingestion

**Chosen: GitHub Actions (cron scheduling)**

| Option | Pros | Cons | Cost |
|---|---|---|---|
| **GitHub Actions** ✅ | Zero infra, built-in logging, retry logic, already in your repo | Not truly real-time | **$0** |
| Apache Kafka | Industry standard, massive throughput, replay-able | Heavy ops overhead, overkill for polling APIs that update every 5–15 mins | ~$20–50/mo |
| AWS Kinesis | Managed Kafka, auto-scaling | AWS lock-in, costs money | Pay per shard |
| Redis Streams | Lightweight, fast | Not ideal for long retention | ~$7/mo |
| Cron on VPS | Simple | Need to manage a server | ~$5/mo |

> City APIs don't update every millisecond. GitHub Actions cron is sufficient and lets you focus on ML rather than infrastructure. Migrate to Kafka later when you understand your data volume — that migration itself becomes a resume talking point.

---

### Analytics Storage

**Chosen: BigQuery**

| Option | Pros | Cons | Cost |
|---|---|---|---|
| **BigQuery** ✅ | Fully managed, massive scale, integrates with every ML tool, SQL-native | Eventual consistency, GCP lock-in | **$0** (10GB storage, 1TB queries/mo free) |
| TimescaleDB | Postgres-compatible, built for time-series | Self-hosted complexity | ~$50/mo cloud |
| InfluxDB | Purpose-built for time-series, fast writes | Custom query language (Flux) | Paid tiers |
| PostgreSQL | Simple, familiar | No time-series optimizations at scale | Server cost |
| DuckDB | Incredibly fast analytics, runs locally | Not great for live writes | $0 but local only |
| Snowflake | Excellent analytics | Trial only, then expensive | $$$ |

> BigQuery's free tier covers 10GB storage and 1TB of queries per month — more than enough for this project indefinitely. "BigQuery" on a resume signals enterprise data infrastructure knowledge.

---

### Data Lake

**Chosen: Cloudflare R2**

| Option | Pros | Cons | Cost |
|---|---|---|---|
| **Cloudflare R2** ✅ | No egress fees, 10GB free forever, S3-compatible API | Newer product, smaller ecosystem | **$0** |
| Google Cloud Storage | Pairs natively with BigQuery, great ecosystem | Egress fees add up | 5GB free, then $0.02/GB |
| AWS S3 | Industry standard, massive ecosystem | Free tier expires after 12 months | 5GB free (12mo), then $0.023/GB |
| Azure Blob | Strong enterprise | Less ML ecosystem friendly | Costly |

> R2's zero egress fees is the deciding factor. Cloud storage costs sneak up via egress, not storage. R2 eliminates that risk entirely.

---

### Pipeline Orchestration

**Chosen: GitHub Actions**

| Option | Pros | Cons | Cost |
|---|---|---|---|
| **GitHub Actions** ✅ | Already in repo, retry logic, dependency steps, full logging | Not purpose-built for data pipelines | **$0** |
| Prefect | Modern, Python-native, great UI | Paid cloud tier starts quickly | $29/mo after free tier |
| Apache Airflow | Industry gold standard, DAG-based, huge community | Heavy self-hosting, steep setup | ~$5–20/mo VPS |
| Dagster | Asset-based thinking, excellent observability | Steeper mental model | Paid quickly |
| Cron jobs | Zero overhead | No retry logic, no visibility | $0 but fragile |

> GitHub Actions handles 80% of what Prefect or Airflow does for this use case at $0. Migrating to Airflow later is straightforward and worth doing as the project matures.

---

### Feature Store

**Chosen: Pandas + Parquet on Cloudflare R2, with Upstash Redis for hot features**

| Option | Pros | Cons | Cost |
|---|---|---|---|
| **Pandas + Parquet** ✅ | Zero infrastructure, portable, immediate | Doesn't scale to millisecond lookups | **$0** |
| **Upstash Redis** ✅ | 10,000 requests/day free, fast cache | Manual feature management, no versioning | **$0** |
| Feast | Open source, production-grade, point-in-time correct joins | Complex setup, premature for v1 | Server cost |
| Tecton | Best-in-class managed feature store | Enterprise-only pricing | $$$ |
| Redis Cloud | Fast, simple | Free tier only 30MB | ~$7/mo |

> A feature store is premature optimization for v1. Build features in Pandas, serialize to Parquet on R2, cache hot features in Upstash Redis. Add Feast when you know exactly what you're computing — then the upgrade is meaningful rather than ceremonial.

---

### Model Training Compute

**Chosen: Kaggle Notebooks (experimentation) + GCP $300 credits (serious training)**

| Option | Pros | Cons | Cost |
|---|---|---|---|
| **Kaggle Notebooks** ✅ | 30 hrs/week free GPU, reliable, no setup | Session limits, not automated | **$0** |
| **GCP Vertex AI** ✅ | Managed training, integrates with BigQuery | Credits expire | **$0** ($300 new account credit) |
| Google Colab | Free T4 GPU | Unreliable, disconnects | $0 free / $10/mo Pro |
| Lambda Labs | Cheapest paid GPU | Costs money | ~$0.50/hr |
| AWS SageMaker | Managed, scalable | Expensive, credits expire | $$$ |
| GitHub Actions | Free CPU | Too slow for LSTM/TFT | $0 |

> Kaggle for iteration, GCP credits for serious training runs. By the time credits run out the project will be mature enough to justify a small spend — or you'll have found a smarter approach.

---

### Experiment Tracking

**Chosen: Weights & Biases**

| Option | Pros | Cons | Cost |
|---|---|---|---|
| **Weights & Biases** ✅ | Best visualizations in ML tooling, unlimited projects, 100GB storage free | Cloud-dependent | **$0** |
| MLflow | Open source, self-hostable, industry standard | Needs hosting, dated UI | Server cost |
| Neptune.ai | Strong metadata management | 100 hours free then paid | Paid quickly |
| Comet ML | Full-featured | Limited free tier | Paid quickly |
| DVC | Git for data + experiments | More complex mental model | $0 |

> W&B's free tier is one of the best deals in ML tooling. The visualizations are impressive enough to screenshot and include in your portfolio and blog post. MLflow is the right corporate answer — W&B makes your project look better to people reviewing your GitHub.

---

### Model Serving

**Chosen: FastAPI + Docker on Google Cloud Run**

| Option | Pros | Cons | Cost |
|---|---|---|---|
| **FastAPI + Cloud Run** ✅ | Auto-generates OpenAPI docs, async, scales to zero, 2M requests/mo free | Python GIL limits at extreme scale | **$0** |
| Flask | Simple, familiar | Synchronous, older patterns | $0 (needs hosting) |
| BentoML | Purpose-built for ML serving | Extra abstraction layer | $0 (needs hosting) |
| TorchServe / TF Serving | Framework-optimized | Framework-locked, heavyweight | $0 (needs hosting) |
| AWS SageMaker | Fully managed | Expensive | $$$ |
| Render free tier | Simple deploy | Cold starts, spins down | $0 with caveats |

> FastAPI appears in nearly every ML engineering job description. Auto-generated docs at `/docs` make it easy to demo. Cloud Run scales to zero when idle — you pay nothing when nobody is using it.

---

### Frontend Framework

**Chosen: Next.js**

| Option | Pros | Cons | Cost |
|---|---|---|---|
| **Next.js** ✅ | React-based, SSR, industry standard, large ecosystem | More complex than plain React for simple needs | **$0** |
| Plain React (Vite) | Simpler, faster to start | No SSR, manual routing | $0 |
| Streamlit | Python, zero frontend knowledge needed | Looks like a homework project, not a product | $0 |
| Vue + Nuxt | Clean syntax | Smaller job market than React | $0 |
| Svelte | Incredibly fast output | Tiny ecosystem for data viz | $0 |

> Streamlit is tempting for speed but signals "data science notebook," not "production product." Next.js is what companies use. The extra complexity proves frontend capability and looks significantly better to hiring managers.

---

### Geospatial Visualization

**Chosen: Deck.gl + MapLibre GL**

| Option | Pros | Cons | Cost |
|---|---|---|---|
| **Deck.gl** ✅ | Built by Uber for large geospatial datasets, open source, stunning output | Steeper learning curve | **$0** |
| **MapLibre GL** ✅ | Open source Mapbox fork, identical API, free forever | Less commercial support | **$0** |
| Mapbox GL JS | Beautiful, performant, highly customizable | Costs after 50,000 loads/mo | $0.50 per 1000 after free tier |
| Google Maps API | Familiar, reliable | Expensive at scale | $$$ |
| Leaflet.js | Simple, well documented | Older, less performant at scale | $0 |
| Kepler.gl | Incredible out-of-the-box visuals | Less flexible for custom apps | $0 |

> MapLibre is literally a fork of Mapbox GL JS with the same API — swap the import and it's identical performance at zero cost. Deck.gl handles large datasets without breaking a sweat. This combo is what Uber, Airbnb, and most serious geospatial products use in production.

---

### Authentication

**Chosen: Clerk**

| Option | Pros | Cons | Cost |
|---|---|---|---|
| **Clerk** ✅ | Best developer experience, beautiful pre-built UI, social login in under an hour | Newer, less enterprise battle-tested | **$0** (10,000 MAU free) |
| Auth0 | Industry standard, very mature | Expensive quickly, complex config | $23/mo after 7,500 MAU |
| Supabase Auth | Unlimited free users, open source | Tied to Supabase ecosystem | **$0** |
| Firebase Auth | Free, Google-backed, simple | Google ecosystem lock-in | **$0** |
| Roll your own JWT | Full control | Security liability, time sink | $0 but never do this |

> You are not building an auth product. Clerk gets you secure, production-grade authentication with social login in under an hour. Use that time on the ML layer instead.

---

### Frontend Hosting

**Chosen: Vercel**

| Option | Pros | Cons | Cost |
|---|---|---|---|
| **Vercel** ✅ | Made by the Next.js team, unlimited personal projects, 100GB bandwidth, instant deploys | Backend/ML workloads need a separate solution | **$0** |
| Netlify | Good free tier | Limited builds per month | $0 with limits |
| Cloudflare Pages | Unlimited bandwidth | Less Next.js native | **$0** |
| GitHub Pages | Free, always on | Static only | **$0** |
| AWS Amplify | AWS ecosystem | Costs money | Paid |

> Vercel and Next.js are made by the same team — it's the canonical deployment target. Personal projects have no meaningful limits on the free tier.

---

### Demo Hosting

**Chosen: Hugging Face Spaces**

| Option | Pros | Cons | Cost |
|---|---|---|---|
| **Hugging Face Spaces** ✅ | Free CPU hosting, respected in ML community, great for portfolio | Public only on free tier | **$0** |
| Streamlit Cloud | Free, simple | Streamlit only | **$0** |
| Render | Simple deploy | Cold starts on free tier | **$0** with caveats |
| Railway | Great DX | $5/mo after credits | Small cost |

> Hugging Face Spaces is where the ML community goes to demo projects. A working demo hosted here carries more weight with ML-focused hiring managers than any other platform.

---

### Full Stack Summary

```
Ingestion         →  GitHub Actions cron                $0
Analytics DB      →  BigQuery                           $0
Data Lake         →  Cloudflare R2                      $0
Orchestration     →  GitHub Actions                     $0
Features          →  Pandas / Parquet + Upstash Redis   $0
Model Training    →  Kaggle + GCP $300 credits          $0
Experiments       →  Weights & Biases                   $0
Model Serving     →  FastAPI + Docker + Cloud Run       $0
Frontend          →  Next.js + Vercel                   $0
Geo Visualization →  Deck.gl + MapLibre GL              $0
Auth              →  Clerk                              $0
ML Demo           →  Hugging Face Spaces                $0
```

**Estimated monthly cost: $0 – $5**

> Free tiers exist because these companies want developers to grow into paying customers. Use them aggressively and without guilt — this is exactly what they are designed for. When this project lands you a role or gains real users, revisit the budget conversation. Until then, every dollar saved is a dollar you don't need to justify.

---

*Last updated: June 2026*
