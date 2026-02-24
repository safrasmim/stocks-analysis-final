import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR

np.random.seed(42)
dates = pd.date_range("2022-01-01", periods=36, freq="MS")
df = pd.DataFrame({
    "date":                 dates,
    "gdp_growth":           np.round(np.random.normal(3.0, 0.5, 36), 2),
    "inflation_rate":       np.round(np.random.normal(2.5, 0.8, 36), 2),
    "interest_rate":        np.round(np.random.normal(5.0, 0.3, 36), 2),
    "unemployment_rate":    np.round(np.random.normal(5.5, 0.4, 36), 2),
    "oil_price_change":     np.round(np.random.normal(0.0, 3.0, 36), 2),
    "currency_rate_change": np.round(np.random.normal(0.0, 0.5, 36), 4),
})
out = DATA_DIR / "macro_indicators.csv"
df.to_csv(out, index=False)
print(f"Macro data saved → {out}")
