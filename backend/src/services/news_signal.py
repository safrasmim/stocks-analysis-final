"""Aggregate news + announcements into a thesis-aligned directional signal."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_ANNOUNCEMENT_WEIGHT = 1.35
DEFAULT_NEWS_WEIGHT = 1.0
DEFAULT_HALF_LIFE_HOURS = 36.0


@dataclass
class AggregateSignalResult:
    ticker: str
    as_of: str
    item_count: int
    probability_up: float
    signal: str
    score: float
    components: dict[str, Any]


def load_ingested_items(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if frame.empty:
        return frame

    frame = frame.copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["published_at", "ticker", "headline"])
    frame["headline"] = frame["headline"].astype(str).str.strip()
    frame = frame[frame["headline"] != ""]
    return frame


def select_recent_items(
    frame: pd.DataFrame,
    *,
    ticker: str,
    as_of: datetime,
    lookback_days: int,
    max_items: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    subset = frame[frame["ticker"].astype(str) == str(ticker)].copy()
    if subset.empty:
        return subset

    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    else:
        as_of_ts = as_of_ts.tz_convert("UTC")
    floor = as_of_ts - pd.Timedelta(days=lookback_days)
    subset = subset[(subset["published_at"] <= as_of_ts) & (subset["published_at"] >= floor)]
    subset = subset.sort_values("published_at", ascending=False)
    if max_items > 0:
        subset = subset.head(max_items)
    return subset


def aggregate_directional_signal(
    predictions: list[dict[str, Any]],
    *,
    as_of: datetime,
    half_life_hours: float = DEFAULT_HALF_LIFE_HOURS,
    announcement_weight: float = DEFAULT_ANNOUNCEMENT_WEIGHT,
    news_weight: float = DEFAULT_NEWS_WEIGHT,
    macro_regime_score: float = 0.0,
) -> AggregateSignalResult:
    if not predictions:
        return AggregateSignalResult(
            ticker="",
            as_of=as_of.date().isoformat(),
            item_count=0,
            probability_up=0.5,
            signal="NEUTRAL",
            score=0.0,
            components={"reason": "No recent items for ticker."},
        )

    weighted_sum = 0.0
    weight_total = 0.0
    bearish_mass = 0.0
    bullish_mass = 0.0

    for row in predictions:
        published_at = pd.Timestamp(row["published_at"])
        if published_at.tzinfo is None:
            published_at = published_at.tz_localize("UTC")
        else:
            published_at = published_at.tz_convert("UTC")

        as_of_ts = pd.Timestamp(as_of)
        if as_of_ts.tzinfo is None:
            as_of_ts = as_of_ts.tz_localize("UTC")
        else:
            as_of_ts = as_of_ts.tz_convert("UTC")

        age_hours = max((as_of_ts - published_at).total_seconds() / 3600.0, 0.0)
        recency_weight = 0.5 ** (age_hours / max(half_life_hours, 1.0))
        type_weight = announcement_weight if row.get("item_type") == "announcement" else news_weight
        base_weight = recency_weight * type_weight

        prob_up = float(row["probability_up"])
        centered = 2.0 * prob_up - 1.0

        weighted_sum += centered * base_weight
        weight_total += base_weight
        bullish_mass += max(centered, 0.0) * base_weight
        bearish_mass += max(-centered, 0.0) * base_weight

    if weight_total == 0:
        score = 0.0
    else:
        score = weighted_sum / weight_total

    macro_adjustment = max(min(macro_regime_score, 1.0), -1.0) * 0.35
    score += macro_adjustment

    if bearish_mass > bullish_mass * 1.2 and score > 0.1:
        score = min(score, 0.1)

    score = max(min(score, 1.0), -1.0)
    probability_up = (score + 1.0) / 2.0

    if score >= 0.15:
        signal = "BULLISH"
    elif score <= -0.15:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    ticker = str(predictions[0].get("ticker", ""))
    return AggregateSignalResult(
        ticker=ticker,
        as_of=as_of.date().isoformat(),
        item_count=len(predictions),
        probability_up=float(round(probability_up, 4)),
        signal=signal,
        score=float(round(score, 4)),
        components={
            "weight_total": round(weight_total, 6),
            "bullish_mass": round(bullish_mass, 6),
            "bearish_mass": round(bearish_mass, 6),
            "macro_adjustment": round(macro_adjustment, 6),
            "half_life_hours": half_life_hours,
            "announcement_weight": announcement_weight,
            "news_weight": news_weight,
        },
    )


def build_prediction_rows(
    *,
    frame: pd.DataFrame,
    predictor: Any,
    ticker: str,
    model: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in frame.itertuples(index=False):
        event_date = pd.Timestamp(item.published_at).date().isoformat()
        pred = predictor.predict(
            texts=[str(item.headline)],
            ticker=ticker,
            model=model,
            event_dates=[event_date],
        )
        rows.append(
            {
                "item_id": str(item.item_id),
                "ticker": str(item.ticker),
                "item_type": str(item.item_type),
                "headline": str(item.headline),
                "published_at": pd.Timestamp(item.published_at).to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
                "probability_up": float(pred["probabilities_up"][0]),
                "prediction": int(pred["predictions"][0]),
            }
        )
    return rows
