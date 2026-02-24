import logging
import re
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)

# ── Saudi / GCC company names ────────────────────────────────────────────────
SAUDI_COMPANIES = {
    "al rajhi", "alrajhi", "sabic", "stc", "aramco", "ncb", "riyad bank",
    "alinma", "sabb", "banque saudi fransi", "bsf", "al jazira", "aljazira",
    "maaden", "masar", "etihad etisalat", "mobily", "zain", "dar al arkan",
    "jarir", "extra", "aldawaa", "nahdi", "panda", "savola", "almarai",
    "saudi electricity", "sec", "acwa power", "neom", "red sea global",
    "sipchem", "yansab", "petrochem", "kemapco", "tasnee",
}

LOCATIONS = {
    "saudi arabia", "riyadh", "jeddah", "mecca", "medina", "dammam",
    "khobar", "jubail", "yanbu", "gcc", "gulf", "uae", "dubai", "abu dhabi",
    "qatar", "kuwait", "bahrain", "oman", "mena", "middle east",
    "us", "usa", "united states", "europe", "china", "asia",
}

MONEY_PATTERNS = [
    r"sar\s*[\d,\.]+\s*(billion|million|trillion)?",
    r"\$\s*[\d,\.]+\s*(billion|million|trillion)?",
    r"[\d,\.]+\s*(billion|million|trillion)\s*(sar|dollar|usd|riyal)",
    r"[\d]+\s*bps",
    r"[\d]+\s*basis\s*points",
]

# ── Event phrase lists ────────────────────────────────────────────────────────
EARNINGS_PHRASES = [
    "profit", "earnings", "net income", "revenue", "eps", "earnings per share",
    "quarterly results", "annual results", "fiscal year", "q1", "q2", "q3", "q4",
    "financial results", "net loss", "operating income", "ebitda", "dividend",
]

POLICY_PHRASES = [
    "central bank", "sama", "federal reserve", "fed ", "ecb", "monetary policy",
    "interest rate", "rate cut", "rate hike", "repo rate", "inflation target",
    "quantitative easing", "quantitative tightening", "dovish", "hawkish",
    "basis points", "bps", "policy rate", "reserve requirement",
]

MERGER_PHRASES = [
    "merger", "acquisition", "takeover", "buyout", "joint venture",
    "partnership", "deal", "agreement", "signed", "mou", "memorandum",
    "stake", "shares acquired", "acquired by", "merges with",
]

INFLATION_PHRASES = [
    "inflation", "cpi", "consumer price", "price index", "cost of living",
    "purchasing power", "deflation", "stagflation", "price rise", "price surge",
]

INTEREST_RATE_PHRASES = [
    "interest rate", "interest rates", "rate cut", "rate hike", "rate reduction",
    "rates cut", "rates hike", "repo rate", "basis points", "bps",
    "federal funds rate", "fed rate", "monetary easing", "monetary tightening",
    "dovish", "hawkish", "yield", "bond yield",
]

# ── NEW: directional macro flags ──────────────────────────────────────────────
RATE_CUT_PHRASES = [
    "rate cut", "rates cut", "interest rate cut", "interest rates cut",
    "rate reduction", "cut rates", "lowered rates", "rates reduced",
    "monetary easing", "quantitative easing", "dovish", "repo rate cut",
    "fed pivot", "bps cut", "basis points cut",
]

RATE_HIKE_PHRASES = [
    "rate hike", "rates hike", "interest rate hike", "interest rates hike",
    "rate increase", "raised rates", "rates raised", "monetary tightening",
    "quantitative tightening", "hawkish", "repo rate hike",
    "bps hike", "basis points hike",
]

OIL_BULLISH_PHRASES = [
    "oil prices rise", "oil price surge", "crude rally", "opec cut",
    "opec+ cut", "opec production cut", "oil demand rises", "brent rises",
]

OIL_BEARISH_PHRASES = [
    "oil prices fall", "oil prices drop", "oil price plunge", "crude selloff",
    "oil demand slow", "oil demand falls", "brent drops", "opec increase",
]


def extract_entity_features(texts: List[str]) -> pd.DataFrame:
    rows = []
    for text in texts:
        t = str(text).lower()
        rows.append({
            "company_mention_count":  sum(1 for c in SAUDI_COMPANIES if c in t),
            "location_mention_count": sum(1 for l in LOCATIONS        if l in t),
            "money_mention_count":    sum(1 for p in MONEY_PATTERNS   if re.search(p, t)),
        })
    return pd.DataFrame(rows)


def extract_event_features(texts: List[str]) -> pd.DataFrame:
    rows = []
    for text in texts:
        t = str(text).lower()
        rows.append({
            # Existing flags
            "is_earnings_news":      int(any(p in t for p in EARNINGS_PHRASES)),
            "is_policy_news":        int(any(p in t for p in POLICY_PHRASES)),
            "is_merger_news":        int(any(p in t for p in MERGER_PHRASES)),
            "is_inflation_news":     int(any(p in t for p in INFLATION_PHRASES)),
            "is_interest_rate_news": int(any(p in t for p in INTEREST_RATE_PHRASES)),
            # NEW directional flags
            "is_rate_cut":           int(any(p in t for p in RATE_CUT_PHRASES)),
            "is_rate_hike":          int(any(p in t for p in RATE_HIKE_PHRASES)),
            "is_oil_bullish":        int(any(p in t for p in OIL_BULLISH_PHRASES)),
            "is_oil_bearish":        int(any(p in t for p in OIL_BEARISH_PHRASES)),
        })
    return pd.DataFrame(rows)
