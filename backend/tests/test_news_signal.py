import sys
from pathlib import Path
from datetime import datetime, timezone
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.news_signal import aggregate_directional_signal, select_recent_items


class NewsSignalTests(unittest.TestCase):
    def test_select_recent_items_filters_window_and_ticker(self):
        frame = pd.DataFrame(
            {
                "ticker": ["2010", "2010", "2222"],
                "headline": ["a", "b", "c"],
                "published_at": pd.to_datetime(
                    [
                        "2026-03-01T08:00:00Z",
                        "2026-03-03T08:00:00Z",
                        "2026-03-03T09:00:00Z",
                    ],
                    utc=True,
                ),
            }
        )
        recent = select_recent_items(
            frame,
            ticker="2010",
            as_of=datetime(2026, 3, 3, 12, 0, tzinfo=timezone.utc),
            lookback_days=2,
            max_items=10,
        )
        self.assertEqual(len(recent), 1)
        self.assertEqual(set(recent["ticker"]), {"2010"})

    def test_aggregate_directional_signal_is_conflict_aware(self):
        preds = [
            {
                "ticker": "2010",
                "item_type": "news",
                "published_at": "2026-03-03T10:00:00+00:00",
                "probability_up": 0.95,
            },
            {
                "ticker": "2010",
                "item_type": "announcement",
                "published_at": "2026-03-03T09:00:00+00:00",
                "probability_up": 0.20,
            },
        ]
        agg = aggregate_directional_signal(
            preds,
            as_of=datetime(2026, 3, 3, 12, 0, tzinfo=timezone.utc),
            macro_regime_score=-0.7,
        )
        self.assertIn(agg.signal, {"BEARISH", "NEUTRAL"})
        self.assertLessEqual(agg.probability_up, 0.55)


if __name__ == "__main__":
    unittest.main()
