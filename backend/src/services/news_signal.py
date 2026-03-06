"""
news_signal.py  —  UPGRADED
Thesis-aligned comprehensive directional signal aggregation.

Upgrades:
 - Category classification (geopolitical > macro_economic > sector_wide > company)
 - Category-based weights: macro/geo news outweighs company news
 - Conflict-aware macro dampener
 - compute_macro_regime_score(): auto-derives regime from news (replaces hard-coded 0.0)
 - AggregateSignalResult extended: macro_dampener, macro_regime, top_drivers, explanation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

CATEGORY_WEIGHTS: dict = {
    "geopolitical":   2.5,
    "macro_economic": 2.0,
    "sector_wide":    1.5,
    "company":        1.0,
    "general":        0.7,
}

DEFAULT_ANNOUNCEMENT_WEIGHT = 1.35
DEFAULT_NEWS_WEIGHT         = 1.0
DEFAULT_HALF_LIFE_HOURS     = 36.0

_GEO_KW = [
    "war","conflict","sanction","military","attack","invasion","geopolit",
    "tension","crisis","embargo","threat","coup","terrorism","escalat",
    "protest","unrest","ceasefire","airstrike","blockade","assassination",
]
_MACRO_KW = [
    "inflation","interest rate","gdp","central bank","fed ","sama",
    "monetary policy","unemployment","oil price","brent","opec",
    "recession","fiscal","trade deficit","currency","forex","budget",
    "stimulus","rate hike","rate cut","cpi","pmi","federal reserve",
    "ecb","imf","world bank","g20","g7","sovereign","yield","bond market",
]
_SECTOR_KW = [
    "sector","industry","regulation","regulatory","ministry","market-wide",
    "all banks","all companies","capital market authority","cma ","tadawul",
    "banking sector","insurance sector",
]


def classify_category(headline: str) -> str:
    """Classify headline into geopolitical | macro_economic | sector_wide | company | general."""
    h = headline.lower()
    if any(k in h for k in _GEO_KW):
        return "geopolitical"
    if any(k in h for k in _MACRO_KW):
        return "macro_economic"
    if any(k in h for k in _SECTOR_KW):
        return "sector_wide"
    return "company"


def compute_macro_regime_score(items_df: pd.DataFrame) -> float:
    """
    Auto-compute macro_regime_score in [-1, 1] from fetched item headlines.
    Replaces the original hard-coded 0.0 default.
    Uses sentiment_compound column if available, else keyword heuristic.
    """
    if items_df.empty:
        return 0.0

    if "category" in items_df.columns:
        macro_items = items_df[items_df["category"].isin(["geopolitical","macro_economic"])]
    else:
        macro_items = items_df[items_df["headline"].apply(
            lambda h: classify_category(str(h)) in ("geopolitical","macro_economic")
        )]

    if macro_items.empty:
        return 0.0

    if "sentiment_compound" in macro_items.columns:
        return float(macro_items["sentiment_compound"].clip(-1, 1).mean())

    neg_kw = ["war","sanction","conflict","recession","inflation surge",
              "rate hike","oil fall","plunge","crisis","attack"]
    pos_kw = ["ceasefire","peace","stimulus","rate cut","oil surge",
              "recovery","growth","gdp rise","trade deal"]
    scores = []
    for h in macro_items["headline"].astype(str):
        hl = h.lower()
        neg = sum(1 for k in neg_kw if k in hl)
        pos = sum(1 for k in pos_kw if k in hl)
        total = max(neg + pos, 1)
        scores.append((pos - neg) / total)
    return float(pd.Series(scores).clip(-1, 1).mean()) if scores else 0.0


@dataclass
class AggregateSignalResult:
    ticker:         str
    as_of:          str
    item_count:     int
    probability_up: float
    signal:         str
    score:          float
    macro_dampener: float                = 0.0
    macro_regime:   float                = 0.0
    components:     dict                 = field(default_factory=dict)
    top_drivers:    list                 = field(default_factory=list)
    explanation:    str                  = ""


def load_ingested_items(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["published_at","ticker","headline"])
    frame["headline"] = frame["headline"].astype(str).str.strip()
    frame = frame[frame["headline"] != ""]
    return frame


def select_recent_items(frame: pd.DataFrame, *, ticker, as_of, lookback_days, max_items) -> pd.DataFrame:
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
    floor  = as_of_ts - pd.Timedelta(days=lookback_days)
    subset = subset[(subset["published_at"] <= as_of_ts) & (subset["published_at"] >= floor)]
    subset = subset.sort_values("published_at", ascending=False)
    if max_items > 0:
        subset = subset.head(max_items)
    return subset


def aggregate_directional_signal(
    predictions: list,
    *,
    as_of: datetime,
    half_life_hours: float     = DEFAULT_HALF_LIFE_HOURS,
    announcement_weight: float = DEFAULT_ANNOUNCEMENT_WEIGHT,
    news_weight: float         = DEFAULT_NEWS_WEIGHT,
    macro_regime_score: float  = 0.0,
) -> AggregateSignalResult:
    if not predictions:
        return AggregateSignalResult(
            ticker="", as_of=as_of.date().isoformat(), item_count=0,
            probability_up=0.5, signal="NEUTRAL", score=0.0,
            explanation="No recent news or announcements for this ticker.",
        )

    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")

    weighted_sum = weight_total = bearish_mass = bullish_mass = macro_neg_pressure = 0.0
    item_details = []

    for row in predictions:
        pub_ts = pd.Timestamp(row["published_at"])
        if pub_ts.tzinfo is None:
            pub_ts = pub_ts.tz_localize("UTC")
        else:
            pub_ts = pub_ts.tz_convert("UTC")

        age_hours = max((as_of_ts - pub_ts).total_seconds() / 3600.0, 0.0)
        recency_w = 0.5 ** (age_hours / max(half_life_hours, 1.0))
        type_w    = announcement_weight if row.get("item_type") == "announcement" else news_weight
        headline  = str(row.get("headline", ""))
        category  = row.get("category") or classify_category(headline)
        cat_w     = CATEGORY_WEIGHTS.get(category, 1.0)
        base_w    = recency_w * type_w * cat_w

        prob_up  = float(row["probability_up"])
        centered = 2.0 * prob_up - 1.0

        weighted_sum       += centered        * base_w
        weight_total       += base_w
        bullish_mass       += max(centered,  0.0) * base_w
        bearish_mass       += max(-centered, 0.0) * base_w

        if category in ("geopolitical","macro_economic") and centered < -0.1:
            macro_neg_pressure += abs(centered) * base_w

        item_details.append({
            "headline":     headline[:80],
            "category":     category,
            "item_type":    row.get("item_type","news"),
            "published_at": str(row.get("published_at","")),
            "probability_up": round(prob_up, 3),
            "weight":       round(base_w, 4),
            "direction":    "UP" if centered > 0 else ("DOWN" if centered < 0 else "NEUTRAL"),
        })

    score = weighted_sum / weight_total if weight_total > 0 else 0.0

    macro_adjustment = max(min(macro_regime_score, 1.0), -1.0) * 0.35
    score += macro_adjustment

    dampener = min(macro_neg_pressure / max(weight_total, 1.0), 0.5)
    if dampener > 0.05 and score > 0.1:
        score = score * (1.0 - dampener * 1.5)

    if bearish_mass > bullish_mass * 1.2 and score > 0.1:
        score = min(score, 0.1)

    score          = max(min(score, 1.0), -1.0)
    probability_up = (score + 1.0) / 2.0
    signal         = "BULLISH" if score >= 0.15 else ("BEARISH" if score <= -0.15 else "NEUTRAL")

    top_drivers = sorted(item_details, key=lambda x: x["weight"], reverse=True)[:5]
    driver_text = "; ".join([
        f"{d['category']} ({d['direction']}): {d['headline'][:50]}"
        for d in top_drivers[:3]
    ])
    macro_note = (
        f" Macro/geopolitical pressure dampened signals by "
        f"{dampener:.0%} (regime: {macro_regime_score:+.2f})."
        if dampener > 0.05 else ""
    )
    explanation = (
        f"{signal} (score: {score:+.2f}, confidence: {min(abs(score)*2,1.0):.0%}). "
        f"Key drivers: {driver_text or 'general market news'}.{macro_note}"
    )

    ticker = str(predictions[0].get("ticker",""))
    return AggregateSignalResult(
        ticker=ticker, as_of=as_of.date().isoformat(), item_count=len(predictions),
        probability_up=float(round(probability_up, 4)), signal=signal,
        score=float(round(score, 4)),
        macro_dampener=float(round(dampener, 4)),
        macro_regime=float(round(macro_regime_score, 4)),
        components={
            "weight_total": round(weight_total, 6), "bullish_mass": round(bullish_mass, 6),
            "bearish_mass": round(bearish_mass, 6), "macro_adjustment": round(macro_adjustment, 6),
            "macro_neg_pressure": round(macro_neg_pressure, 6), "dampener": round(dampener, 4),
            "half_life_hours": half_life_hours, "announcement_weight": announcement_weight,
            "news_weight": news_weight,
        },
        top_drivers=top_drivers, explanation=explanation,
    )


def build_prediction_rows(*, frame: pd.DataFrame, predictor, ticker: str, model: str) -> list:
    rows = []
    for item in frame.itertuples(index=False):
        event_date = pd.Timestamp(item.published_at).date().isoformat()
        headline   = str(item.headline)
        pred = predictor.predict(texts=[headline], ticker=ticker, model=model, event_dates=[event_date])
        rows.append({
            "item_id":       str(item.item_id),
            "ticker":        str(item.ticker),
            "item_type":     str(item.item_type),
            "headline":      headline,
            "category":      classify_category(headline),
            "published_at":  pd.Timestamp(item.published_at).to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
            "probability_up": float(pred["probabilities_up"][0]),
            "prediction":    int(pred["predictions"][0]),
        })
    return rows
