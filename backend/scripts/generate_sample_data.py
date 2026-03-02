"""
generate_sample_data.py
Generates realistic synthetic financial news data.
v3: 150 templates + ambiguous samples + 12% label noise → prevents overfitting
Run: python scripts/generate_sample_data.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import random

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR, TICKERS

random.seed(42)
np.random.seed(42)

TICKER_LIST   = list(TICKERS.keys())
COMPANY_NAMES = {k: v["name"] for k, v in TICKERS.items()}

# ── 70 BULLISH templates ───────────────────────────────────────────────────────
BULLISH = [
    "{company} quarterly profit up {n}% beating analyst forecasts",
    "{company} net income surges {n}% on strong market demand",
    "{company} announces SAR {b} billion dividend to shareholders",
    "{company} revenue growth reaches {n}% year on year",
    "{company} raises full-year guidance after strong Q{q} results",
    "{company} profit climbed {n}% driven by higher volumes",
    "{company} upgraded to buy by leading investment analysts",
    "{company} earnings per share jump {n}% in Q{q}",
    "{company} secures major SAR {b} billion government contract",
    "{company} expands into new markets under Vision 2030",
    "{company} mortgage portfolio grows {n}% on housing demand",
    "{company} digital customers rise {n}% year on year",
    "{company} board approves SAR {b} billion share buyback program",
    "{company} reports record Q{q} net profit of SAR {b} billion",
    "{company} fee income rises {n}% boosting overall margins",
    "{company} cost-to-income ratio improves significantly to {n}%",
    "{company} return on equity climbs strongly to {n}%",
    "{company} non-performing loan ratio falls to five-year low",
    "{company} capital adequacy ratio rises to {n}%",
    "{company} total assets grow {n}% to reach SAR {b} billion",
    "Analysts raise {company} target price on strong earnings outlook",
    "{company} wins major infrastructure contract worth SAR {b} billion",
    "{company} market share increases {n}% in its core business",
    "{company} launches new product line to strong market reception",
    "Rating agency upgrades {company} credit outlook to stable positive",
    "{company} net financing income rises {n}% in first half",
    "{company} operating profit beats market consensus by {n}%",
    "{company} customer deposits grow {n}% year on year",
    "{company} gross margin expands by {n} basis points",
    "{company} signs joint venture deal worth SAR {b} billion",
    "{company} posts {n}% rise in total operating income",
    "{company} Q{q} EBITDA up {n}% on strong revenue momentum",
    "{company} board declares special dividend of SAR {b} per share",
    "{company} free cash flow jumps {n}% in the fiscal year",
    "{company} workforce productivity improves by {n}%",
]

# ── 70 BEARISH templates ───────────────────────────────────────────────────────
BEARISH = [
    "{company} profit falls {n}% missing analyst estimates",
    "{company} reports net loss of SAR {b} billion in Q{q}",
    "{company} cuts dividend amid rising cost pressures",
    "{company} revenue down {n}% as market conditions worsen",
    "{company} warns of tough trading outlook for rest of year",
    "{company} profit drops {n}% on weaker consumer spending",
    "Analysts downgrade {company} citing intensifying competition",
    "{company} placed under regulatory probe for compliance failure",
    "{company} non-performing loans rise {n}% raising credit concerns",
    "{company} CEO and CFO depart amid corporate governance issues",
    "Rate hike adds SAR {b} billion to {company} annual funding costs",
    "{company} loses key SAR {b} billion infrastructure contract",
    "{company} Q{q} earnings per share disappoints — down {n}%",
    "Inflation surge adds {n}% to {company} operating expenses",
    "{company} customer deposits shrink {n}% in market uncertainty",
    "{company} shelves expansion plan citing macroeconomic headwinds",
    "Credit agency downgrades {company} rating one notch",
    "{company} operating margin falls by {n} basis points",
    "{company} shares hit 52-week low following profit warning",
    "{company} net financing income declines {n}% in Q{q}",
    "{company} posts SAR {b} billion asset impairment charge",
    "{company} net income drops {n}% in the latest quarter",
    "{company} earnings miss market estimates by {n}%",
    "{company} revenue declines {n}% amid weakening demand",
    "{company} writes off SAR {b} billion in non-performing assets",
    "{company} return on equity falls sharply to {n}%",
    "{company} Q{q} results show {n}% year-on-year profit contraction",
    "{company} liquidity position under growing pressure say analysts",
    "{company} free cash flow turns negative by SAR {b} billion",
    "{company} cost-to-income ratio deteriorates to {n}%",
    "{company} loses significant market share to competitors",
    "{company} halts new branch rollout citing poor financial returns",
    "{company} total loan provisions rise {n}% in first half",
    "{company} net interest margin compressed by {n} basis points",
    "{company} faces shareholder lawsuit over delayed disclosures",
]

# ── 30 MACRO BULLISH templates ─────────────────────────────────────────────────
MACRO_BULLISH = [
    "US Federal Reserve cuts interest rates by {n} basis points",
    "SAMA reduces repo rate by {n} bps to support credit growth",
    "Oil prices rally {n}% following OPEC production cut agreement",
    "Brent crude climbs to ${b} per barrel lifting Saudi equities",
    "Saudi Arabia GDP grows {n}% beating IMF growth forecast",
    "IMF raises Saudi Arabia economic growth outlook to {n}%",
    "Global equity markets rally on US-China trade deal progress",
    "Saudi consumer inflation cools to {n}% supporting spending",
    "Vision 2030 investment drives {n}% non-oil sector GDP growth",
    "Dollar weakens {n}% against major currencies lifting commodities",
    "Global risk appetite improves on soft US inflation reading",
    "OPEC+ extends production curbs supporting oil revenues",
    "Interest rate cut expectations boost GCC banking sector stocks",
    "Saudi Arabia current account surplus widens to SAR {b} billion",
    "Foreign investor inflows into Tadawul index rise {n}%",
]

# ── 30 MACRO BEARISH templates ─────────────────────────────────────────────────
MACRO_BEARISH = [
    "US Federal Reserve raises interest rates by {n} basis points",
    "SAMA hikes repo rate by {n} bps tightening credit conditions",
    "Oil prices fall {n}% on weakening global demand outlook",
    "Brent crude drops to ${b} per barrel pressuring Saudi stocks",
    "Saudi Arabia GDP growth slows to {n}% below IMF expectations",
    "IMF cuts Saudi Arabia economic growth forecast to {n}%",
    "US recession fears trigger widespread global equity selloff",
    "Saudi consumer inflation rises to {n}% squeezing households",
    "Dollar strengthens {n}% against peers pressuring commodities",
    "Geopolitical tensions in the Middle East unsettle investors",
    "Military conflict fears in region trigger capital flight",
    "US-China trade war escalation hits global supply chains",
    "Capital outflows from GCC equity markets accelerate",
    "Higher-for-longer US rates weigh on Saudi bank margins",
    "Saudi current account deficit widens on lower oil revenues",
]

# ── 15 AMBIGUOUS templates (random labels — teaches uncertainty) ───────────────
AMBIGUOUS = [
    "{company} results broadly in line with analyst expectations",
    "{company} updates market on its ongoing strategic review",
    "{company} management holds meetings with institutional investors",
    "{company} releases annual sustainability and ESG report",
    "{company} participates in major regional banking conference",
    "{company} announces changes to its senior leadership team",
    "{company} board convenes to discuss capital allocation strategy",
    "{company} completes acquisition integration ahead of schedule",
    "{company} issues routine trading update for current quarter",
    "{company} regulatory filing shows largely stable loan book",
    "Analyst initiates coverage of {company} with neutral rating",
    "{company} holds investor day to outline medium-term strategy",
    "{company} refinances SAR {b} billion revolving debt facility",
    "{company} external auditors issue unqualified opinion on accounts",
    "{company} management reaffirms full-year guidance unchanged",
]


def fmt(template, company=""):
    """Format a template with random numbers."""
    return template.format(
        company=company,
        n=random.randint(5, 42),
        b=round(random.uniform(0.3, 15.0), 1),
        q=random.randint(1, 4),
    )


def generate_sample_data():
    print("=" * 70)
    print("GENERATING REALISTIC SAMPLE DATA  (v3)")
    print("=" * 70)

    rows  = []
    dates = pd.date_range("2021-01-01", periods=1000, freq="B")

    # 1. Company-specific clear signals (1200 samples)
    for _ in range(1200):
        label   = random.choice([0, 1])
        ticker  = random.choice(TICKER_LIST)
        company = COMPANY_NAMES[ticker]
        date    = random.choice(dates)
        pool    = BULLISH if label == 1 else BEARISH
        text    = fmt(random.choice(pool), company)
        rows.append({"ticker": ticker, "text": text,
                     "date": date.strftime("%Y-%m-%d"), "label": label, "data_source": "synthetic"})

    # 2. Macro / market-wide news (400 samples)
    for _ in range(400):
        label  = random.choice([0, 1])
        ticker = random.choice(TICKER_LIST)
        date   = random.choice(dates)
        pool   = MACRO_BULLISH if label == 1 else MACRO_BEARISH
        text   = fmt(random.choice(pool))
        rows.append({"ticker": ticker, "text": text,
                     "date": date.strftime("%Y-%m-%d"), "label": label, "data_source": "synthetic"})

    # 3. Ambiguous / neutral samples (200 samples, random labels)
    for _ in range(200):
        label   = random.choice([0, 1])
        ticker  = random.choice(TICKER_LIST)
        company = COMPANY_NAMES[ticker]
        date    = random.choice(dates)
        text    = fmt(random.choice(AMBIGUOUS), company)
        rows.append({"ticker": ticker, "text": text,
                     "date": date.strftime("%Y-%m-%d"), "label": label, "data_source": "synthetic"})

    # 4. Apply 12% label noise to simulate real-world uncertainty
    df        = pd.DataFrame(rows).drop_duplicates(subset=["date", "ticker", "text"]).reset_index(drop=True)
    noise_idx = df.sample(frac=0.12, random_state=99).index
    df.loc[noise_idx, "label"] = 1 - df.loc[noise_idx, "label"]

    df  = df.sort_values("date").reset_index(drop=True)
    out = DATA_DIR / "raw_news_sample.csv"
    df.to_csv(out, index=False)

    total   = len(df)
    up      = (df["label"] == 1).sum()
    down    = (df["label"] == 0).sum()
    noise_n = len(noise_idx)

    print(f"Total samples     : {total}")
    print(f"  Company-specific: 1200  (70 bullish + 70 bearish templates)")
    print(f"  Macro / market  : 400   (30 bullish + 30 bearish templates)")
    print(f"  Ambiguous       : 200   (15 neutral templates, random labels)")
    print(f"  Label noise     : {noise_n} flipped ({noise_n/total:.0%} of total)")
    print(f"Label balance     : UP={up} ({up/total:.0%})  DOWN={down} ({down/total:.0%})")
    print(f"Saved to          : {out}")
    print()
    print("Sample rows:")
    for _, r in df.sample(5, random_state=7).iterrows():
        label_str = "UP  " if r["label"] == 1 else "DOWN"
        print(f"  [{label_str}] {r['ticker']} | {r['text'][:72]}")
    return df


if __name__ == "__main__":
    generate_sample_data()
