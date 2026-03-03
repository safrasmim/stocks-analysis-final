"""Incremental TDWL news + announcement ingestion from Mubasher mix2."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, TICKERS
from src.ingestion.mubasher_tdwl import (
    MubasherRequestConfig,
    REQUEST_TYPE_ANNOUNCEMENT,
    REQUEST_TYPE_NEWS,
    build_mix2_url,
    compute_latest_cursor,
    fetch_mix2_json,
    filter_items_newer_than_cursor,
    load_cursor,
    parse_mix2_items,
    save_cursor,
    upsert_incremental_items,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=sorted(TICKERS.keys()),
        help="Subset of TDWL tickers to ingest (default: all configured tickers)",
    )
    parser.add_argument("--sid", default="sid", help="Mubasher SID query value")
    parser.add_argument("--uid", default="123", help="Mubasher UID query value")
    parser.add_argument("--language", default="EN", help="Language query value, e.g. EN or AR")
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "ingested" / "tdwl_news_announcements.parquet"),
        help="Output dataset path (.parquet or .csv)",
    )
    parser.add_argument(
        "--cursor",
        default=str(DATA_DIR / "ingested" / "tdwl_news_announcements.cursor.json"),
        help="Cursor file path used for incremental fetch",
    )
    parser.add_argument(
        "--extra-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra query params forwarded to mix2 URL, can be repeated",
    )
    return parser.parse_args()


def _parse_extra_params(raw_pairs: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for pair in raw_pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid --extra-param format '{pair}', expected KEY=VALUE")
        key, value = pair.split("=", 1)
        params[key.strip()] = value.strip()
    return params


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    allowed_tickers = set(args.tickers)
    unknown = sorted(allowed_tickers - set(TICKERS.keys()))
    if unknown:
        raise SystemExit(f"Unknown ticker(s): {', '.join(unknown)}")

    request_config = MubasherRequestConfig(sid=args.sid, uid=args.uid, language=args.language)
    output_path = Path(args.output)
    cursor_path = Path(args.cursor)
    cursor = load_cursor(cursor_path)
    extra_params = _parse_extra_params(args.extra_param)

    logging.info("Fetching TDWL announcements/news for %s tickers", len(allowed_tickers))

    all_items = []
    for item_type, rt in (("announcement", REQUEST_TYPE_ANNOUNCEMENT), ("news", REQUEST_TYPE_NEWS)):
        url = build_mix2_url(rt=rt, config=request_config, extra_params=extra_params)
        logging.info("Calling RT=%s URL: %s", rt, url)

        payload = fetch_mix2_json(url)
        parsed = parse_mix2_items(payload, item_type=item_type, allowed_tickers=allowed_tickers)
        fresh = filter_items_newer_than_cursor(parsed, cursor=cursor)

        logging.info(
            "%s records from endpoint, %s newer than cursor",
            len(parsed),
            len(fresh),
        )
        all_items.extend(fresh)

    dataset = upsert_incremental_items(all_items, store_path=output_path)
    new_cursor = compute_latest_cursor(dataset)
    save_cursor(new_cursor, cursor_path)

    logging.info("Stored %s total records in %s", len(dataset), output_path)
    logging.info("Cursor saved to %s", cursor_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
