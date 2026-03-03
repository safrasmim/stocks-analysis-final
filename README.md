# stocks-analysis-final

Thesis project: **Predicting Stock Price Movements through Macroeconomic Variables and Financial News**.

## Core workflows

### 1) Historical baseline (OHLCV-only)
```bash
python backend/scripts/train_historical_baseline.py
```

### 2) News + macro model training
```bash
python backend/scripts/train_all_models.py --dataset backend/data/raw_news_sample.csv --allow-synthetic --time-aware
```

### 3) Held-out evaluation
```bash
python backend/scripts/evaluate_all_models.py
```

### 4) Walk-forward evaluation
```bash
python backend/scripts/walk_forward_evaluate_news.py --dataset backend/data/raw_news_sample.csv --allow-synthetic --n-folds 5
```

### 5) Ablation study
```bash
python backend/scripts/run_ablation_study.py --dataset backend/data/raw_news_sample.csv --allow-synthetic
```

### 6) Incremental TDWL news + announcement ingestion
```bash
python backend/scripts/fetch_tdwl_news_announcements.py --sid <sid> --uid <uid>
```
Stores merged records in `backend/data/ingested/tdwl_news_announcements.parquet` and a cursor file so only newer dates are processed on later runs.

### 7) Live API workflows (news ingestion + aggregate ticker signal)
```bash
# Trigger ingestion from API (POST JSON body)
curl -X POST http://127.0.0.1:8000/ingestion/tdwl/run \
  -H "Content-Type: application/json" \
  -d '{"sid":"<sid>","uid":"<uid>","tickers":["2010","2222"]}'

# Get final combined ticker signal from latest items (not one-news only)
curl -X POST http://127.0.0.1:8000/predict/ticker/live \
  -H "Content-Type: application/json" \
  -d '{"ticker":"2010","lookback_days":3,"max_items":50,"macro_regime_score":-0.3}'
```
This endpoint aggregates multiple recent news + announcements with recency weighting and macro-regime adjustment to produce one final BULLISH/NEUTRAL/BEARISH output.

See `THESIS_ALIGNMENT_REMEDIATION.md` for thesis-readiness details.


### Windows one-click demo
Use:
```bat
backend\run_command_backend.bat
```
This script generates synthetic data and therefore trains with `--allow-synthetic --time-aware` automatically.
