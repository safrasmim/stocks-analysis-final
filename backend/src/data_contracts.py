"""Data validation helpers for thesis-grade reproducibility."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

REQUIRED_NEWS_COLUMNS = ["date", "ticker", "text", "label", "data_source"]
REQUIRED_STOCK_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "ticker", "label"]
REQUIRED_MACRO_COLUMNS = [
    "date",
    "gdp_growth",
    "inflation_rate",
    "interest_rate",
    "unemployment_rate",
    "oil_price_change",
    "currency_rate_change",
]
ALLOWED_SOURCES = {"real", "synthetic", "mixed"}


@dataclass
class ValidationResult:
    rows: int
    sources: list[str]


def _assert_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def validate_news_dataframe(df: pd.DataFrame, *, allow_synthetic: bool = False) -> ValidationResult:
    """Validate news dataset and guard against accidental synthetic experiments."""
    _assert_columns(df, REQUIRED_NEWS_COLUMNS)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Dataset contains empty text rows.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Dataset has invalid date values.")

    duplicate_count = int(df.duplicated(subset=["date", "ticker", "text"]).sum())
    if duplicate_count:
        raise ValueError(f"Dataset contains {duplicate_count} duplicate (date,ticker,text) rows.")

    sources = sorted(set(df["data_source"].astype(str).str.lower()))
    invalid_sources = [s for s in sources if s not in ALLOWED_SOURCES]
    if invalid_sources:
        raise ValueError(f"Unsupported data_source values: {invalid_sources}")

    if not allow_synthetic:
        banned = {"synthetic", "mixed"}
        if any(s in banned for s in sources):
            raise ValueError(
                "Synthetic data detected in research mode. "
                "Use --allow-synthetic only for demo development runs."
            )

    return ValidationResult(rows=len(df), sources=sources)


def validate_stock_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Validate OHLCV stock dataset for historical baseline experiments."""
    _assert_columns(df, REQUIRED_STOCK_COLUMNS)
    if df.empty:
        raise ValueError("Stock dataset is empty.")

    frame = df.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if frame["Date"].isna().any():
        raise ValueError("Stock dataset has invalid Date values.")

    if frame[["Open", "High", "Low", "Close", "Volume"]].isna().any().any():
        raise ValueError("Stock dataset has null OHLCV values.")

    dup = int(frame.duplicated(subset=["Date", "ticker"]).sum())
    if dup:
        raise ValueError(f"Stock dataset has {dup} duplicate (Date,ticker) rows.")

    return ValidationResult(rows=len(frame), sources=sorted(set(frame.get("source", "unknown").astype(str))))


def validate_macro_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Validate macro indicators dataset."""
    _assert_columns(df, REQUIRED_MACRO_COLUMNS)
    if df.empty:
        raise ValueError("Macro dataset is empty.")

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("Macro dataset has invalid date values.")

    if frame[REQUIRED_MACRO_COLUMNS[1:]].isna().any().any():
        raise ValueError("Macro dataset has null indicator values.")

    return ValidationResult(rows=len(frame), sources=["macro"])
