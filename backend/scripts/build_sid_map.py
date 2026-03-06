"""
build_sid_map.py  --  find Mubasher SID for each of our 8 tickers
Reads TSID from the ANNL mix2 response and saves a JSON mapping.
Run once: python scripts/build_sid_map.py
"""
import json, time
from pathlib import Path
from urllib.parse import urlencode
import requests

TICKERS = ["1120","2010","7010","1150","4325","2222","1211","4110"]
OUT     = Path("data/processed/sid_map.json")

# Date windows to search — go back further if ticker not found
WINDOWS = [
    ("20260201000000","20260305000000"),
    ("20260101000000","20260201000000"),
    ("20251101000000","20260101000000"),
    ("20250901000000","20251101000000"),
    ("20250601000000","20250901000000"),
    ("20250101000000","20250601000000"),
]


def fetch_annl(sd, ed, session):
    params = {
        "SID":"sid","UID":"123","RT":25,"E":"TDWL","UE":"TDWL",
        "L":"EN","AE":1,"UNC":0,"M":1,"H":1,
        "SD":sd,"ED":ed,
    }
    url = "https://data-sa9.mubasher.net/mix2?" + urlencode(params)
    try:
        r = session.get(url, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("  fetch error:", e)
        return None


def extract_tsids(payload):
    """Return dict of ticker -> list of TSIDs from ANNL response."""
    result = {}
    hed = (payload.get("HED") or {}).get("ANNL") or ""
    recs = (payload.get("DAT") or {}).get("ANNL") or []
    if not hed or not recs:
        return result
    cols = hed.split("|")
    ci   = {c:i for i,c in enumerate(cols)}
    i_s    = ci.get("S")
    i_tsid = ci.get("TSID")
    i_si   = ci.get("SI")
    if i_s is None:
        return result
    for rec in recs:
        parts = rec.split("|")
        tkr   = parts[i_s].strip() if i_s < len(parts) else ""
        if tkr not in TICKERS:
            continue
        tsids = []
        if i_tsid is not None and i_tsid < len(parts):
            v = parts[i_tsid].strip()
            if v:
                tsids += [x.strip() for x in v.split(",") if x.strip()]
        if i_si is not None and i_si < len(parts):
            v = parts[i_si].strip()
            if v:
                tsids += [x.strip() for x in v.split(",") if x.strip()]
        if tkr not in result:
            result[tkr] = []
        for sid in tsids:
            if sid and sid not in result[tkr]:
                result[tkr].append(sid)
    return result


def main():
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    mapping = {}

    for sd, ed in WINDOWS:
        still_needed = [t for t in TICKERS if t not in mapping]
        if not still_needed:
            break
        print("Searching", sd[:8], "->", ed[:8], "  need:", still_needed)
        payload = fetch_annl(sd, ed, session)
        if not payload:
            time.sleep(3)
            continue
        found = extract_tsids(payload)
        for tkr, sids in found.items():
            if tkr not in mapping and sids:
                mapping[tkr] = sids
                print("  FOUND", tkr, "->", sids)
        time.sleep(1)

    print()
    print("=" * 50)
    print("MAPPING RESULT:")
    for t in TICKERS:
        sids = mapping.get(t, [])
        status = "OK  -> " + str(sids) if sids else "MISSING"
        print("  " + t.ljust(6) + status)
    missing = [t for t in TICKERS if t not in mapping]
    if missing:
        print("WARNING: no SID found for:", missing)
    print("=" * 50)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(mapping, indent=2))
    print("Saved to:", OUT)
    print()
    print("Next step:")
    print("  python scripts/fetch_mubasher_ohlc.py --start 2025-09-01")


if __name__ == "__main__":
    main()
