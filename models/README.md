# models/

Training scripts, saved model artifacts, and evaluation code. Experiments are tracked in Weights & Biases (see root `CLAUDE.md`) — this folder holds the code and final artifacts, not experiment logs.

One subfolder per domain, owned by that domain's teammate:

- `transit/` — ridership demand forecasting (Prophet → XGBoost → TFT)
- `energy/` — grid load forecasting
- `weather/` — weather forecasting + anomaly detection (feeds anomaly flags to the other two domains)

Saved model weights/checkpoints should not be committed to git if large — use W&B Artifacts or R2, and gitignore the binary files.
