"""
fetch_mubasher_ohlc.py  --  download + parse Mubasher OHLC ZIPs
Requires data/processed/sid_map.json (run build_sid_map.py first)

Usage:
    python scripts/fetch_mubasher_ohlc.py --start 2025-09-01
"""
import argparse, io, json, logging, time, zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests

FILES_URL = "https://data-sa9.mubasher.net/pro_plus_archives/trades/tdwl/files.txt"
ZIP_URL   = "https://data-sa9.mubasher.net/pro_plus_archives/ohlc/tdwl/{d}.zip"
SID_MAP   = Path("data/processed/sid_map.json")
OUT_PATH  = Path("data/processed/ohlc_tdwl.csv")
TIMEOUT   = 60
LOG = logging.getLogger(__name__)

# CSV columns inside each ZIP file
# INS|TMIN|OP|HIG|LOW|CLS|VOL|TOVR|NOTR|VWAP|...|CHG|PCHG|MCAP|...
HDR = "INS|TMIN|OP|HIG|LOW|CLS|VOL|TOVR|NOTR|VWAP|CIT|CIV|CITR|COT|COV|COTR|CHG|PCHG|MCAP|PER|PBR|PSR|PCR|DYR|LAST|CNV|CNT|PFCF|DY|TBQ|DA|TAQ|UTV|BBP|BAP|DTV"


def http_get(url, session, binary=False):
    for n in range(1, 4):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content if binary else r.text
        except requests.RequestException as e:
            LOG.warning("attempt %d/3: %s", n, e)
            time.sleep(2 ** n)
    return None


def available_dates(session):
    txt = http_get(FILES_URL, session)
    if not txt:
        return []
    out = []
    for ln in txt.splitlines():
        p = ln.strip().split(",")[0].replace(".zip","").strip()
        if p.isdigit() and len(p) == 8:
            out.append(p)
    LOG.info("files.txt: %d dates", len(out))
    return sorted(out)


def parse_csv_to_daily(raw_text, ticker, trade_date):
    """Aggregate minute-bar CSV into one daily OHLC row."""
    lines = raw_text.strip().splitlines()
    if len(lines) < 2:
        return None
    header = lines[0].split("|")
    ci = {c:i for i,c in enumerate(header)}
    op_list, hi_list, lo_list, cl_list, vol_list = [], [], [], [], []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        p = ln.split("|")
        def gf(name):
            ix = ci.get(name)
            if ix is None or ix >= len(p):
                return None
            try: return float(p[ix])
            except: return None
        op = gf("OP"); hi = gf("HIG"); lo = gf("LOW")
        cl = gf("CLS"); vo = gf("VOL")
        if op: op_list.append(op)
        if hi: hi_list.append(hi)
        if lo: lo_list.append(lo)
        if cl: cl_list.append(cl)
        if vo: vol_list.append(vo)
    if not cl_list:
        return None
    return {
        "date":     trade_date,
        "ticker":   ticker,
        "open":     op_list[0]  if op_list else None,
        "high":     max(hi_list) if hi_list else None,
        "low":      min(lo_list) if lo_list else None,
        "close":    cl_list[-1] if cl_list else None,
        "volume":   sum(vol_list) if vol_list else None,
        "bars":     len(cl_list),
    }


def fetch_zip(date_str, session, sid_to_ticker):
    url     = ZIP_URL.format(d=date_str)
    content = http_get(url, session, binary=True)
    if not content:
        return []
    rows = []
    trade_date = date_str[:4]+"-"+date_str[4:6]+"-"+date_str[6:]
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for fname in zf.namelist():
                sid = fname.replace(".csv","").strip()
                tkr = sid_to_ticker.get(sid)
                if not tkr:
                    continue
                raw = zf.read(fname).decode("utf-8", errors="replace")
                row = parse_csv_to_daily(raw, tkr, trade_date)
                if row:
                    rows.append(row)
    except zipfile.BadZipFile:
        LOG.warning("bad zip: %s", date_str)
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--start",   default="2025-09-01")
    p.add_argument("--end",     default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--output",  default=str(OUT_PATH))
    p.add_argument("--sleep",   type=float, default=0.3)
    a = p.parse_args()

    if not SID_MAP.exists():
        print("ERROR: sid_map.json not found. Run first:")
        print("  python scripts/build_sid_map.py")
        return 1

    sid_map = json.loads(SID_MAP.read_text())
    # Invert: SID -> ticker (each ticker may have multiple SIDs)
    sid_to_ticker = {}
    for tkr, sids in sid_map.items():
        for sid in sids:
            sid_to_ticker[sid] = tkr
    LOG.info("SID map: %d entries for %d tickers", len(sid_to_ticker), len(sid_map))

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"

    s_dt   = datetime.strptime(a.start, "%Y-%m-%d")
    e_dt   = datetime.strptime(a.end,   "%Y-%m-%d")
    avail  = set(available_dates(session))
    wanted = []
    cur = s_dt
    while cur <= e_dt:
        ds = cur.strftime("%Y%m%d")
        if ds in avail:
            wanted.append(ds)
        cur += timedelta(days=1)
    LOG.info("%d ZIP files to download", len(wanted))

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        ed   = pd.read_csv(out, usecols=["date"])
        done = set(pd.to_datetime(ed["date"]).dt.strftime("%Y%m%d").tolist())
    todo = [d for d in wanted if d not in done]
    LOG.info("Fetching %d new dates...", len(todo))

    new_rows = []
    for i, ds in enumerate(todo, 1):
        r = fetch_zip(ds, session, sid_to_ticker)
        new_rows.extend(r)
        LOG.info("[%d/%d] %s -> %d rows", i, len(todo), ds, len(r))
        time.sleep(a.sleep)

    # TO (only fail if ALL dates returned 0):
    if not new_rows and len(todo) > 3:
        LOG.warning("No rows fetched. Check sid_map.json has correct SIDs.")
        return 1
    LOG.info("No new rows this run (holiday or already up to date).")
    return 0

    ndf = pd.DataFrame(new_rows)
    combined = pd.concat([pd.read_csv(out), ndf], ignore_index=True) if out.exists() else ndf
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.drop_duplicates(subset=["date","ticker"]).sort_values(["ticker","date"]).reset_index(drop=True)
    combined.to_csv(out, index=False)

    print("=" * 55)
    print("SAVED  : " + str(out))
    print("ROWS   : " + str(len(combined)))
    print("RANGE  : " + str(combined.date.min().date()) + " -> " + str(combined.date.max().date()))
    print("TICKERS:")
    for t, n in combined.ticker.value_counts().items():
        print("  " + str(t).ljust(8) + str(n) + " days")
    print("=" * 55)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
