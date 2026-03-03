"""Build an aggregate signal report for one ticker from ingested TDWL items."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, MODELS_DIR, TICKERS
from src.predictor import Predictor
from src.services.news_signal import (
    aggregate_directional_signal,
    build_prediction_rows,
    load_ingested_items,
    select_recent_items,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--model", default="ensemble")
    parser.add_argument("--lookback-days", type=int, default=3)
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--macro-regime-score", type=float, default=0.0)
    parser.add_argument(
        "--ingestion-path",
        default=str(DATA_DIR / "ingested" / "tdwl_news_announcements.parquet"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ticker not in TICKERS:
        raise SystemExit(f"Unknown ticker: {args.ticker}")

    predictor = Predictor(model_dir=MODELS_DIR)
    predictor.load()

    frame = load_ingested_items(Path(args.ingestion_path))
    as_of = datetime.now(timezone.utc)
    recent = select_recent_items(
        frame,
        ticker=args.ticker,
        as_of=as_of,
        lookback_days=max(args.lookback_days, 1),
        max_items=max(args.max_items, 1),
    )
    rows = build_prediction_rows(frame=recent, predictor=predictor, ticker=args.ticker, model=args.model) if not recent.empty else []
    aggregate = aggregate_directional_signal(rows, as_of=as_of, macro_regime_score=args.macro_regime_score)

    print({
        "ticker": args.ticker,
        "signal": aggregate.signal,
        "probability_up": aggregate.probability_up,
        "score": aggregate.score,
        "items_used": aggregate.item_count,
        "components": aggregate.components,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
