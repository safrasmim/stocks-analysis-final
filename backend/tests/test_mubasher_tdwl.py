import sys
from pathlib import Path
from datetime import datetime, timezone
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.mubasher_tdwl import (
    ParsedItem,
    compute_latest_cursor,
    filter_items_newer_than_cursor,
    parse_mix2_items,
)


class MubasherTdwlTests(unittest.TestCase):
    def test_parse_mix2_items_filters_tickers_and_parses_date(self) -> None:
        payload = {
            "HED": {"NWSL": ["ID", "E", "S", "HED", "DT", "L", "PRV"]},
            "DAT": {
                "NWSL": [
                    "A1|TDWL|2010|SABIC reports earnings|20260210153045|EN|MUB",
                    "A2|TDWL|9999|Other symbol|20260211120000|EN|MUB",
                ]
            },
        }

        items = parse_mix2_items(payload, item_type="news", allowed_tickers={"2010"})

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].ticker, "2010")
        self.assertEqual(items[0].published_at, datetime(2026, 2, 10, 15, 30, 45, tzinfo=timezone.utc))

    def test_cursor_filtering_keeps_only_newer_items(self) -> None:
        item_old = ParsedItem("1", "news", "TDWL", "2010", "MUB", "EN", "old", datetime(2026, 1, 1, tzinfo=timezone.utc), "20260101")
        item_new = ParsedItem("2", "news", "TDWL", "2010", "MUB", "EN", "new", datetime(2026, 1, 2, tzinfo=timezone.utc), "20260102")

        cursor = {"news:2010": "2026-01-01T00:00:00+00:00"}
        filtered = filter_items_newer_than_cursor([item_old, item_new], cursor=cursor)

        self.assertEqual([i.item_id for i in filtered], ["2"])

    def test_compute_latest_cursor(self) -> None:
        frame = pd.DataFrame(
            {
                "item_type": ["news", "news", "announcement"],
                "ticker": ["2010", "2010", "2010"],
                "published_at": pd.to_datetime([
                    "2026-01-01T00:00:00Z",
                    "2026-01-03T00:00:00Z",
                    "2026-01-02T00:00:00Z",
                ], utc=True),
            }
        )

        cursor = compute_latest_cursor(frame)

        self.assertTrue(cursor["news:2010"].startswith("2026-01-03"))
        self.assertTrue(cursor["announcement:2010"].startswith("2026-01-02"))


if __name__ == "__main__":
    unittest.main()
