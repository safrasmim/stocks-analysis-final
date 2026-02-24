import logging
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path

from .text_stats    import extract_text_stats
from .sentiment     import extract_sentiment_features
from .topics        import extract_topic_features
from .entities      import extract_entity_features, extract_event_features
from .macro         import add_macro_features
from .sector_impact import get_sector_score   # NEW

logger = logging.getLogger(__name__)

ALL_FEATURE_COLS = [
    # Text statistics
    "text_length", "word_count", "avg_word_length",
    "sentence_count", "exclamation_count", "question_count",
    # Sentiment (FinBERT / rule-based + finance overrides)
    "sentiment_negative", "sentiment_neutral",
    "sentiment_positive", "sentiment_compound",
    # LDA topic distribution
    *[f"topic_{i}" for i in range(10)],
    # Named entity counts
    "company_mention_count", "location_mention_count", "money_mention_count",
    # Event type flags (original)
    "is_earnings_news", "is_policy_news", "is_merger_news",
    "is_inflation_news", "is_interest_rate_news",
    # Directional event flags
    "is_rate_cut", "is_rate_hike",
    "is_oil_bullish", "is_oil_bearish",
    # Macro sector features (NEW)
    "sector_sentiment_score",   # float -1.0 to +1.0
    "is_macro_news",            # 1 if macro event detected
    # Macro economic indicators
    "gdp_growth", "inflation_rate", "interest_rate",
    "unemployment_rate", "oil_price_change", "currency_rate_change",
]


def extract_all_features(
    df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    lda_model=None,
    lda_dict=None,
    model_dir: Optional[Path] = None,
) -> pd.DataFrame:
    df    = df.copy().reset_index(drop=True)
    texts = df["text"].tolist()
    logger.info("Extracting features for %d texts...", len(texts))

    df = pd.concat([
        df,
        extract_text_stats(texts),
        extract_sentiment_features(texts),
        extract_topic_features(texts, lda_model, lda_dict, model_dir=model_dir),
        extract_entity_features(texts),
        extract_event_features(texts),
    ], axis=1)

    # Macro / sector features (NEW)
    sector_rows = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", ""))
        text   = str(row.get("text",   ""))
        info   = get_sector_score(text, ticker)
        sector_rows.append({
            "sector_sentiment_score": info["sector_sentiment_score"],
            "is_macro_news":          float(info["is_macro_news"]),
        })
    df = pd.concat([df, pd.DataFrame(sector_rows, index=df.index)], axis=1)

    # Macro economic indicators
    if macro_df is not None:
        df = add_macro_features(df, macro_df)
    else:
        for col in ["gdp_growth", "inflation_rate", "interest_rate",
                    "unemployment_rate", "oil_price_change", "currency_rate_change"]:
            df[col] = 0.0

    return df


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    return df[cols].fillna(0).values.astype(np.float32)
