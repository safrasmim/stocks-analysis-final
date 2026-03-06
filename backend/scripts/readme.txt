# Retrain only (data already fresh)
python scripts/run_pipeline.py --skip-ohlc --skip-news

# Rebuild dataset + retrain, no new downloads
python scripts/run_pipeline.py --skip-ohlc --skip-news

# Try different label strategy
python scripts/run_pipeline.py --skip-ohlc --skip-news --label same_day

# Extend history further back
python scripts/run_pipeline.py --start 2025-07-01


Every future run is one command:

powershell
python scripts/run_pipeline.py          # full refresh with new data
python scripts/run_pipeline.py --skip-ohlc --skip-news   # retrain only

To document the regime transfer finding separately:

powershell
python scripts/train_all_models.py --split-mode time


Your Two Commands Going Forward
powershell
# Weekly refresh — new data + retrain (≈97s)
python scripts/run_pipeline.py

# Thesis regime experiment — time-aware split
python scripts/train_all_models.py --split-mode time
# then: python scripts/evaluate_all_models.py
# Remember to restore: python scripts/run_pipeline.py --skip-ohlc --skip-news