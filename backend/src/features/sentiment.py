import logging
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)

FINANCE_SENTIMENT_OVERRIDES = {
    # Rate policy — BULLISH
    "rate cut":            0.6,
    "rates cut":           0.6,
    "interest rate cut":   0.7,
    "interest rates cut":  0.7,
    "rate reduction":      0.6,
    "monetary easing":     0.7,
    "quantitative easing": 0.6,
    "dovish":              0.6,
    "repo rate cut":       0.7,
    "fed pivot":           0.65,
    "cut rates":           0.6,
    "lowered rates":       0.6,
    # Rate policy — BEARISH
    "rate hike":               -0.6,
    "rates hike":              -0.6,
    "interest rate hike":      -0.7,
    "interest rates hike":     -0.7,
    "rate increase":           -0.5,
    "rates raised":            -0.6,
    "monetary tightening":     -0.7,
    "quantitative tightening": -0.6,
    "hawkish":                 -0.6,
    "repo rate hike":          -0.7,
    # Oil — BULLISH
    "oil prices rise":  0.5,
    "oil price surge":  0.6,
    "crude rally":      0.6,
    "opec cut":         0.5,
    "opec+ cut":        0.5,
    # Oil — BEARISH
    "oil prices fall":  -0.5,
    "oil price plunge": -0.6,
    "crude selloff":    -0.6,
    "oil demand slow":  -0.5,
}


def apply_finance_overrides(text: str, compound: float) -> float:
    """Override FinBERT/rule-based score for macro phrases it misclassifies."""
    text_lower = text.lower()
    for phrase, score in FINANCE_SENTIMENT_OVERRIDES.items():
        if phrase in text_lower:
            return score
    return compound


def extract_sentiment_features(texts: List[str], batch_size: int = 16) -> pd.DataFrame:
    try:
        from transformers import pipeline
        pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )
        results = []
        for i in range(0, len(texts), batch_size):
            batch = [str(t)[:512] for t in texts[i:i + batch_size]]
            for text, output in zip(batch, pipe(batch)):        # ← text added
                scores   = {item["label"]: item["score"] for item in output}
                neg      = scores.get("negative", 0.0)
                neu      = scores.get("neutral",  0.0)
                pos      = scores.get("positive", 0.0)
                compound = apply_finance_overrides(text, pos - neg)  # ← CALLED HERE
                if compound != pos - neg:          # override fired — adjust pos/neg
                    if compound > 0:
                        pos, neg = compound, 0.0
                        neu = max(0.0, 1.0 - pos)
                    else:
                        pos, neg = 0.0, abs(compound)
                        neu = max(0.0, 1.0 - neg)
                results.append({
                    "sentiment_negative": neg,
                    "sentiment_neutral":  neu,
                    "sentiment_positive": pos,
                    "sentiment_compound": compound,
                })
        return pd.DataFrame(results)

    except Exception as e:
        logger.warning("FinBERT failed (%s). Using rule-based fallback.", e)
        return _rule_based_sentiment(texts)


def _rule_based_sentiment(texts: List[str]) -> pd.DataFrame:
    pos_words = {
        "profit", "growth", "increase", "surge", "gain", "rise", "strong",
        "record", "beat", "exceed", "positive", "up", "rally", "boost",
        "upgrade", "dividend", "expand", "success", "high",
    }
    # ⚠️ "cut" REMOVED — it's ambiguous ("rate cut" = bullish, handled by overrides)
    neg_words = {
        "loss", "decline", "decrease", "fall", "drop", "weak", "miss",
        "below", "negative", "concern", "risk", "down", "hike", "plunge",
        "downgrade", "resign", "probe", "impairment",
    }
    rows = []
    for text in texts:
        words    = str(text).lower().split()
        pos      = sum(1 for w in words if w in pos_words)
        neg      = sum(1 for w in words if w in neg_words)
        total    = max(pos + neg, 1)
        p, n     = pos / total, neg / total
        compound = apply_finance_overrides(text, p - n)   # ← CALLED HERE
        if compound != p - n:
            if compound > 0:
                p, n = compound, 0.0
            else:
                p, n = 0.0, abs(compound)
        rows.append({
            "sentiment_negative": n,
            "sentiment_neutral":  max(1 - p - n, 0),
            "sentiment_positive": p,
            "sentiment_compound": compound,
        })
    return pd.DataFrame(rows)
