"""
find_all_sids.py  --  find SIDs for all 8 tickers using RT=25 and RT=27
Queries one ticker at a time to avoid timeouts.
Run: python scripts/find_all_sids.py
"""
import json, time, pathlib
from urllib.parse import urlencode
import requests

TICKERS = ["1120","2010","7010","1150","4325","2222","1211","4110"]
OUT     = pathlib.Path("data/processed/sid_map.json")

# Seed with what we already know from RT=27 searches
KNOWN = {
    "2010": ["14224","13834210322"],
    "2222": ["885456","142149","14296","14272"],
    "7010": ["142121","13834210377"],
    "1120": ["14210","13834210321"],
    "4325": ["13854977809","13854977810"],
    "1211": ["14214","13834210439"],
    "1150": ["14212","13834210393"],
}

# Search windows (small = no timeout)
WINDOWS = [
    ("20260201000000","20260305000000"),
    ("20260101000000","20260201000000"),
    ("20251101000000","20260101000000"),
    ("20251001000000","20251101000000"),
    ("20250901000000","20251001000000"),
    ("20250801000000","20250901000000"),
    ("20250601000000","20250801000000"),
    ("20250401000000","20250601000000"),
    ("20250101000000","20250401000000"),
    ("20240701000000","20250101000000"),
    ("20240101000000","20240701000000"),
]


def fetch(session, rt, sd, ed, ticker=None):
    params = {
        "SID":"sid","UID":"123","RT":rt,
        "E":"TDWL","UE":"TDWL","L":"EN",
        "AE":1,"UNC":0,"M":1,"H":1,
        "SD":sd,"ED":ed,
    }
    if ticker:
        params["S"] = ticker  # filter by specific ticker
    url = "https://data-sa9.mubasher.net/mix2?" + urlencode(params)
    try:
        r = session.get(url, timeout=45)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None


def extract_sids(payload, target_ticker):
    """Extract TSIDs for a specific ticker from NWSL or ANNL response."""
    if not payload:
        return []
    hed_sec = payload.get("HED") or {}
    dat_sec = payload.get("DAT") or {}
    data_key = "NWSL" if "NWSL" in hed_sec else "ANNL" if "ANNL" in hed_sec else None
    if not data_key:
        return []
    header = hed_sec[data_key].split("|")
    ci     = {c:i for i,c in enumerate(header)}
    i_s    = ci.get("S")
    i_tsid = ci.get("TSID")
    i_si   = ci.get("SI")
    if i_s is None:
        return []
    tsids = []
    for rec in dat_sec.get(data_key, []):
        parts  = rec.split("|")
        ticker = parts[i_s].split(",")[0].strip() if i_s < len(parts) else ""
        if ticker != target_ticker:
            continue
        for ix in [i_tsid, i_si]:
            if ix is not None and ix < len(parts):
                v = parts[ix].strip()
                if v:
                    tsids += [x.strip() for x in v.split(",") if x.strip()]
    return list(dict.fromkeys(tsids))  # deduplicated, order preserved


def main():
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    found = dict(KNOWN)  # start with known SIDs

    missing = [t for t in TICKERS if t not in found]
    print("Starting with", len(found), "known,", len(missing), "missing:", missing)
    print()

    for ticker in TICKERS:
        if ticker in found:
            print(ticker, "-> already known:", found[ticker][:3])
            continue
        print("Searching for", ticker, "...")
        for sd, ed in WINDOWS:
            # Try RT=25 (announcements) with ticker filter FIRST — more results
            for rt in [25, 27]:
                payload = fetch(session, rt, sd, ed, ticker=ticker)
                sids    = extract_sids(payload, ticker)
                if sids:
                    found[ticker] = sids
                    print("  FOUND via RT=" + str(rt) + " [" + sd[:8] + "] -> " + str(sids[:4]))
                    break
                time.sleep(0.3)
            if ticker in found:
                break
            time.sleep(0.5)
        if ticker not in found:
            print("  STILL MISSING after all windows")

    # Clean: keep short SIDs (<=6 digits) first, then long ones, max 4 total
    clean = {}
    for tkr, sids in found.items():
        short = [x for x in sids if len(x) <= 6  and x.isdigit()]
        long_ = [x for x in sids if len(x)  > 6  and x.isdigit()]
        clean[tkr] = (short + long_)[:6]

    print()
    print("=" * 55)
    print("FINAL SID MAP:")
    for t in TICKERS:
        sids = clean.get(t, [])
        tag  = "OK " if sids else "MISSING"
        print("  " + t.ljust(6) + tag + " " + str(sids))
    print("=" * 55)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(clean, indent=2))
    print("Saved ->", OUT)
    print()
    print("Next: python scripts/fetch_mubasher_ohlc.py --start 2025-09-01")


if __name__ == "__main__":
    main()
