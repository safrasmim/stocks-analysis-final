# Thesis Alignment Remediation Plan (Repo-Specific)

This file now tracks implemented and pending actions for strong thesis-defense readiness.

## A) Implemented in this repo

### 1) Data integrity and provenance controls
- `backend/src/data_contracts.py`
  - validates news schema/date/duplicates/source tags
  - validates stock OHLCV dataset schema for historical baseline
  - validates macro indicators schema
- synthetic rows are explicitly tagged in generated sample data (`data_source: synthetic`).

### 2) Training reproducibility controls
- `backend/scripts/train_all_models.py`
  - supports dataset argument and time-aware split
  - explicit synthetic override flag
  - emits `models/experiment_manifest.json`

### 3) New true historical baseline pipeline (thesis critical)
- `backend/scripts/train_historical_baseline.py`
  - uses only OHLCV + technical indicators (RSI, MACD, rolling volatility, lag returns, SMA features)
  - trains a separate baseline artifact in `models/historical_baseline/`
  - outputs baseline metrics json for fair model comparison

### 4) New walk-forward evaluation tooling (thesis critical)
- `backend/scripts/walk_forward_evaluate_news.py`
  - chronology-safe fold-by-fold evaluation
  - reports per-fold RF / Gradient Boosting / Ensemble
  - exports mean/std/worst metrics to `data/evaluation/walk_forward_metrics.json`

### 5) New ablation study tooling (thesis critical)
- `backend/scripts/run_ablation_study.py`
  - full model vs no_sentiment vs no_topics vs no_entities_events vs no_macro
  - exports `data/evaluation/ablation_results.json`

## B) Remaining items before final thesis submission

1. Replace synthetic `raw_news_sample.csv` with real curated dataset and source provenance table.
2. Add confidence calibration + abstention policy curves.
3. Add regime-based robustness slices (high-volatility / policy-shock windows).
4. Add drift monitoring report automation.

## C) Recommended command flow for thesis experiments

```bash
# 0) compile/sanity
python -m compileall backend/src backend/scripts

# 1) train historical-price baseline
python backend/scripts/train_historical_baseline.py

# 2) train macro-news models (use real dataset in final thesis)
python backend/scripts/train_all_models.py --dataset backend/data/raw_news_sample.csv --allow-synthetic --time-aware

# 3) standard held-out metrics
python backend/scripts/evaluate_all_models.py

# 4) walk-forward backtesting report
python backend/scripts/walk_forward_evaluate_news.py --dataset backend/data/raw_news_sample.csv --allow-synthetic --n-folds 5

# 5) ablation report
python backend/scripts/run_ablation_study.py --dataset backend/data/raw_news_sample.csv --allow-synthetic
```
