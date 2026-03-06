"""
fetch_tdwl_news_announcements.py  —  UPGRADED

Modes:
  Incremental (default):
    python scripts/fetch_tdwl_news_announcements.py

  Full history backfill:
    python scripts/fetch_tdwl_news_announcements.py --backfill-from 2025-01-01

  Custom date window:
    python scripts/fetch_tdwl_news_announcements.py --start 2026-01-01 --end 2026-03-01

  Specific tickers:
    python scripts/fetch_tdwl_news_announcements.py --tickers 1120 2010 2222
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, TICKERS
from src.ingestion.mubasher_tdwl import (
    ANN_CHUNK_DAYS, NEWS_CHUNK_DAYS,
    MubasherRequestConfig,
    REQUEST_TYPE_ANNOUNCEMENT, REQUEST_TYPE_NEWS,
    compute_latest_cursor, fetch_date_range, fetch_latest,
    filter_items_newer_than_cursor, load_cursor, save_cursor,
    upsert_incremental_items,
)

DEFAULT_OUTPUT = str(DATA_DIR / "ingested" / "tdwl_news_announcements.parquet")
DEFAULT_CURSOR = str(DATA_DIR / "ingested" / "tdwl_news_announcements.cursor.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers",  nargs="*", default=sorted(TICKERS.keys()))
    p.add_argument("--sid",      default="sid")
    p.add_argument("--uid",      default="123")
    p.add_argument("--language", default="EN")
    p.add_argument("--output",   default=DEFAULT_OUTPUT)
    p.add_argument("--cursor",   default=DEFAULT_CURSOR)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--backfill-from", metavar="YYYY-MM-DD")
    mode.add_argument("--start",         metavar="YYYY-MM-DD")
    p.add_argument("--end", metavar="YYYY-MM-DD")
    p.add_argument("--extra-param", action="append", default=[], metavar="KEY=VALUE")
    return p.parse_args()


def _parse_extra(raw):
    params = {}
    for pair in raw:
        if "=" not in pair:
            raise ValueError(f"Bad --extra-param: '{pair}'. Expected KEY=VALUE")
        k, v = pair.split("=", 1)
        params[k.strip()] = v.strip()
    return params


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    allowed_tickers = set(args.tickers)
    unknown = sorted(allowed_tickers - set(TICKERS.keys()))
    if unknown:
        raise SystemExit(f"Unknown ticker(s): {', '.join(unknown)}")

    config      = MubasherRequestConfig(sid=args.sid, uid=args.uid, language=args.language)
    output_path = Path(args.output)
    cursor_path = Path(args.cursor)
    cursor      = load_cursor(cursor_path)
    now         = datetime.now(timezone.utc)
    all_items   = []

    if args.backfill_from:
        start_dt = datetime.strptime(args.backfill_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        logging.info("BACKFILL MODE: %s -> today for %d tickers", args.backfill_from, len(allowed_tickers))
        for item_type, rt, chunk in (
            ("news",         REQUEST_TYPE_NEWS,         NEWS_CHUNK_DAYS),
            ("announcement", REQUEST_TYPE_ANNOUNCEMENT, ANN_CHUNK_DAYS),
        ):
            all_items.extend(fetch_date_range(
                rt=rt, config=config, start_dt=start_dt, end_dt=now,
                item_type=item_type, allowed_tickers=allowed_tickers, chunk_days=chunk,
            ))

    elif args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.end else now
        logging.info("RANGE MODE: %s -> %s", args.start, end_dt.strftime("%Y-%m-%d"))
        for item_type, rt, chunk in (
            ("news",         REQUEST_TYPE_NEWS,         NEWS_CHUNK_DAYS),
            ("announcement", REQUEST_TYPE_ANNOUNCEMENT, ANN_CHUNK_DAYS),
        ):
            all_items.extend(fetch_date_range(
                rt=rt, config=config, start_dt=start_dt, end_dt=end_dt,
                item_type=item_type, allowed_tickers=allowed_tickers, chunk_days=chunk,
            ))

    else:
        logging.info("INCREMENTAL MODE: latest window, cursor-filtered")
        for item_type, rt, chunk in (
            ("news",         REQUEST_TYPE_NEWS,         NEWS_CHUNK_DAYS),
            ("announcement", REQUEST_TYPE_ANNOUNCEMENT, ANN_CHUNK_DAYS),
        ):
            items = fetch_latest(rt=rt, config=config, item_type=item_type,
                                 allowed_tickers=allowed_tickers, chunk_days=chunk)
            fresh = filter_items_newer_than_cursor(items, cursor=cursor)
            logging.info("%s: %d fetched, %d newer than cursor", item_type, len(items), len(fresh))
            all_items.extend(fresh)

    dataset    = upsert_incremental_items(all_items, store_path=output_path)
    new_cursor = compute_latest_cursor(dataset)
    save_cursor(new_cursor, cursor_path)
    logging.info("Done. %d new | %d total | cursor saved to %s", len(all_items), len(dataset), cursor_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
