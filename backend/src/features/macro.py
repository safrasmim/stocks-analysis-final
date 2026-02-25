import pandas as pd
import numpy as np
from pathlib import Path

MACRO_COLS = ["gdp_growth","inflation_rate","interest_rate","unemployment_rate","oil_price_change","currency_rate_change"]

def load_macro_data(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        return df
    return _generate_synthetic_macro()

def _generate_synthetic_macro() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=36, freq="MS")
    np.random.seed(42)
    return pd.DataFrame({
        "date":                 dates,
        "gdp_growth":           np.random.normal(3.0, 0.5, 36),
        "inflation_rate":       np.random.normal(2.5, 0.8, 36),
        "interest_rate":        np.random.normal(5.0, 0.3, 36),
        "unemployment_rate":    np.random.normal(5.5, 0.4, 36),
        "oil_price_change":     np.random.normal(0.0, 3.0, 36),
        "currency_rate_change": np.random.normal(0.0, 0.5, 36),
    })

def add_macro_features(df: pd.DataFrame, macro_df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    macro_df = macro_df.copy()
    macro_df["date"] = pd.to_datetime(macro_df["date"], errors="coerce")

    # Use as-of join so inference can use the latest available macro snapshot
    # instead of defaulting to zeros when exact month values are missing.
    df["asof_date"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    macro_df["asof_date"] = macro_df["date"].dt.to_period("M").dt.to_timestamp()

    df = df.sort_values("asof_date")
    macro_df = macro_df.sort_values("asof_date")

    macro_cols = ["asof_date", *MACRO_COLS]
    df = pd.merge_asof(
        df,
        macro_df[macro_cols],
        on="asof_date",
        direction="backward",
    ).drop(columns=["asof_date"])

    for col in MACRO_COLS:
        if col not in df.columns: df[col] = 0.0
        df[col] = df[col].fillna(0)
    return df
