"""
Macro Event Classifier — detects market-moving macro events in headlines
"""

MACRO_EVENTS = {
    "war_middle_east": [
        "war", "conflict", "military", "attack", "iran", "strike",
        "missile", "armed", "combat", "invasion", "troops", "airstrikes",
        "sanctions", "blockade", "crisis", "escalation",
    ],
    "oil_price_rise": [
        "oil prices rise", "oil price surge", "crude rally", "brent rises",
        "opec cut", "opec production cut", "oil demand rises",
    ],
    "oil_price_fall": [
        "oil prices fall", "oil price plunge", "crude selloff", "brent drops",
        "opec increase", "oil demand slow", "oil glut",
    ],
    "rate_cut": [
        "rate cut", "rates cut", "interest rate cut", "monetary easing",
        "dovish", "fed pivot", "repo rate cut", "lowered rates",
    ],
    "rate_hike": [
        "rate hike", "rates raised", "monetary tightening", "hawkish",
        "interest rate hike", "repo rate hike",
    ],
    "us_recession": [
        "recession", "economic slowdown", "gdp contraction",
        "global recession", "economic downturn", "stagflation",
    ],
    "geopolitical_risk": [
        "geopolitical", "political instability", "sanctions",
        "trade war", "tension", "uncertainty", "embargo",
    ],
    "gdp_growth": [
        "gdp growth", "economic expansion", "economic growth",
        "imf raises", "growth forecast", "economy expands",
    ],
    "usd_strengthen": [
        "dollar strengthens", "dollar rises", "usd rally",
        "strong dollar", "dollar index rises",
    ],
}


def detect_macro_events(headline: str) -> dict:
    """Return dict of detected macro events for a headline."""
    t        = headline.lower()
    detected = {}
    for event, keywords in MACRO_EVENTS.items():
        if any(kw in t for kw in keywords):
            detected[event] = True
    return detected


def is_macro_news(headline: str) -> bool:
    """True if headline contains any macro event."""
    return len(detect_macro_events(headline)) > 0
