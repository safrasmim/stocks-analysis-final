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

See `THESIS_ALIGNMENT_REMEDIATION.md` for thesis-readiness details.
