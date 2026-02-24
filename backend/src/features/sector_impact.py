"""
Sector Impact Engine — maps macro events to per-sector sentiment scores
"""
from .macro_events import detect_macro_events

# Sensitivity of each sector to each macro event (-1.0 to +1.0)
SECTOR_SENSITIVITY = {
    "Banking": {
        "war_middle_east":   -0.80,
        "oil_price_rise":    +0.40,
        "oil_price_fall":    -0.40,
        "rate_cut":          +0.60,
        "rate_hike":         -0.50,
        "us_recession":      -0.60,
        "geopolitical_risk": -0.70,
        "gdp_growth":        +0.50,
        "usd_strengthen":    -0.30,
    },
    "Petrochemicals": {
        "war_middle_east":   -0.60,
        "oil_price_rise":    +0.70,
        "oil_price_fall":    -0.50,
        "rate_cut":          +0.30,
        "rate_hike":         -0.30,
        "us_recession":      -0.70,
        "geopolitical_risk": -0.50,
        "gdp_growth":        +0.60,
        "usd_strengthen":    +0.40,
    },
    "Energy": {
        "war_middle_east":   -0.20,
        "oil_price_rise":    +0.90,
        "oil_price_fall":    -0.90,
        "rate_cut":          +0.20,
        "rate_hike":         -0.20,
        "us_recession":      -0.60,
        "geopolitical_risk": -0.10,
        "gdp_growth":        +0.50,
        "usd_strengthen":    +0.50,
    },
    "Telecom": {
        "war_middle_east":   -0.30,
        "oil_price_rise":    +0.20,
        "oil_price_fall":    -0.10,
        "rate_cut":          +0.30,
        "rate_hike":         -0.20,
        "us_recession":      -0.20,
        "geopolitical_risk": -0.20,
        "gdp_growth":        +0.30,
        "usd_strengthen":    -0.10,
    },
    "Real_Estate": {
        "war_middle_east":   -0.70,
        "oil_price_rise":    +0.50,
        "oil_price_fall":    -0.40,
        "rate_cut":          +0.70,
        "rate_hike":         -0.80,
        "us_recession":      -0.50,
        "geopolitical_risk": -0.60,
        "gdp_growth":        +0.60,
        "usd_strengthen":    -0.20,
    },
    "Retail_Consumer": {
        "war_middle_east":   -0.60,
        "oil_price_rise":    +0.30,
        "oil_price_fall":    -0.20,
        "rate_cut":          +0.40,
        "rate_hike":         -0.40,
        "us_recession":      -0.40,
        "geopolitical_risk": -0.50,
        "gdp_growth":        +0.70,
        "usd_strengthen":    -0.30,
    },
    "Industry": {
        "war_middle_east":   -0.50,
        "oil_price_rise":    +0.40,
        "oil_price_fall":    -0.30,
        "rate_cut":          +0.30,
        "rate_hike":         -0.30,
        "us_recession":      -0.60,
        "geopolitical_risk": -0.40,
        "gdp_growth":        +0.60,
        "usd_strengthen":    +0.20,
    },
}

# Ticker → Sector mapping (50 stocks)
TICKER_SECTOR = {
    # Banking
    "1120":"Banking","1150":"Banking","1050":"Banking","1060":"Banking",
    "1080":"Banking","1090":"Banking","1100":"Banking","1110":"Banking",
    "1010":"Banking","1020":"Banking",
    # Petrochemicals
    "2010":"Petrochemicals","2020":"Petrochemicals","2060":"Petrochemicals",
    "2070":"Petrochemicals","2080":"Petrochemicals","2090":"Petrochemicals",
    "2130":"Petrochemicals","2150":"Petrochemicals","2170":"Petrochemicals",
    "2210":"Petrochemicals",
    # Energy
    "2222":"Energy","2350":"Energy","5110":"Energy","5210":"Energy","1820":"Energy",
    # Telecom
    "7010":"Telecom","7020":"Telecom","7030":"Telecom",
    # Real Estate
    "4020":"Real_Estate","4030":"Real_Estate","4090":"Real_Estate","4100":"Real_Estate",
    "4150":"Real_Estate","4160":"Real_Estate","4200":"Real_Estate","4325":"Real_Estate",
    # Retail
    "4190":"Retail_Consumer","4230":"Retail_Consumer","4240":"Retail_Consumer",
    "4260":"Retail_Consumer","4290":"Retail_Consumer","4050":"Retail_Consumer",
    "4210":"Retail_Consumer","4080":"Retail_Consumer",
    # Industry
    "1214":"Industry","2110":"Industry","2250":"Industry",
    "4140":"Industry","4170":"Industry","2040":"Industry",
}

# Ticker → Company name
TICKER_NAME = {
    "1120":"Al Rajhi Bank","1150":"Alinma Bank","1050":"NCB","1060":"Riyad Bank",
    "1080":"Arab National Bank","1090":"Saudi Investment Bank","1100":"Banque Saudi Fransi",
    "1110":"Al Jazira Bank","1010":"Al Bilad Bank","1020":"SABB",
    "2010":"SABIC","2020":"Yansab","2060":"Petrochem","2070":"Sipchem",
    "2080":"Tasnee","2090":"Ibn Rushd","2130":"Advanced Petrochem",
    "2150":"National Industrialization","2170":"SAFCO","2210":"Nama Chem",
    "2222":"Saudi Aramco","2350":"Saudi Aramco","5110":"ACWA Power",
    "5210":"Saudi Electricity","1820":"ENGIE Saudi",
    "7010":"STC","7020":"Mobily","7030":"Zain Saudi",
    "4020":"Dar Al Arkan","4030":"Emaar EC","4090":"Arriyadh Dev",
    "4100":"KEC","4150":"Jabal Omar","4160":"Makkah Const",
    "4200":"Taiba","4325":"MASAR",
    "4190":"Jarir","4230":"Extra","4240":"Nahdi","4260":"Al Dawaa",
    "4290":"Aldrees","4050":"Savola","4210":"Almarai","4080":"Panda",
    "1214":"Maaden","2110":"Saudi Kayan","2250":"Saudi Industrial",
    "4140":"Al Hassan Ghazi","4170":"Saudi Cable","2040":"National Metal",
}


def get_sector_score(headline: str, ticker: str) -> dict:
    """Compute sector-adjusted sentiment score for a headline + ticker."""
    sector  = TICKER_SECTOR.get(ticker, "Unknown")
    events  = detect_macro_events(headline)
    sens    = SECTOR_SENSITIVITY.get(sector, {})
    score   = sum(sens.get(e, 0.0) for e in events)
    # Clamp to [-1, 1]
    score   = max(-1.0, min(1.0, score))
    return {
        "sector":                sector,
        "sector_sentiment_score": score,
        "is_macro_news":          int(len(events) > 0),
        "detected_events":        list(events.keys()),
    }


def broadcast_to_all_tickers(headline: str) -> list:
    """Run macro prediction for all 50 tickers from one headline."""
    results = []
    for ticker, sector in TICKER_SECTOR.items():
        info  = get_sector_score(headline, ticker)
        score = info["sector_sentiment_score"]
        results.append({
            "ticker":    ticker,
            "name":      TICKER_NAME.get(ticker, ticker),
            "sector":    sector,
            "score":     score,
            "direction": "UP" if score > 0.1 else ("DOWN" if score < -0.1 else "NEUTRAL"),
            "confidence": round(abs(score) * 100, 1),
            "events":    info["detected_events"],
        })
    # Sort by score ascending (most bearish first)
    return sorted(results, key=lambda x: x["score"])
