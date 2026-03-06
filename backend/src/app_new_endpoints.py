"""
app_new_endpoints.py
====================
New / upgraded endpoints — paste into (or import from) your existing app.py.

Steps:
  A) Replace LiveTickerSignalResponse class in app.py with the one below.
  B) Add BackfillRequest class next to the other Pydantic models.
  C) Add the decorator lines (currently commented out) back, then paste
     all the async functions at the bottom of app.py.
  D) The predict_live_ticker_signal_v2 fully replaces the original.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel


# ── Updated Pydantic model (replace existing LiveTickerSignalResponse) ────────

class LiveTickerSignalResponse(BaseModel):
    ticker:         str
    company:        str
    as_of:          str
    signal:         str
    probability_up: float
    score:          float
    items_used:     int
    macro_dampener: float                = 0.0
    macro_regime:   float                = 0.0
    explanation:    str                  = ""
    top_drivers:    List[Dict[str, Any]] = []
    components:     Dict[str, Any]       = {}
    items:          List[Dict[str, Any]] = []


class BackfillRequest(BaseModel):
    tickers:       Optional[List[str]] = None
    backfill_from: str                 = "2025-01-01"
    sid:           str                 = "sid"
    uid:           str                 = "123"
    output_path:   Optional[str]       = None
    cursor_path:   Optional[str]       = None


# ── POST /ingestion/tdwl/backfill ─────────────────────────────────────────────
# @app.post("/ingestion/tdwl/backfill", response_model=IngestionRunResponse, tags=["Ingestion"])
async def run_tdwl_backfill(request: BackfillRequest):
    """One-time history backfill using chunked date windows. Run once on deployment."""
    from src.ingestion.mubasher_tdwl import (
        ANN_CHUNK_DAYS, NEWS_CHUNK_DAYS, MubasherRequestConfig,
        REQUEST_TYPE_ANNOUNCEMENT, REQUEST_TYPE_NEWS,
        compute_latest_cursor, fetch_date_range, save_cursor, upsert_incremental_items,
    )
    from src.services.news_signal import load_ingested_items

    tickers = set(request.tickers or list(TICKERS.keys()))
    unknown = sorted(tickers - set(TICKERS.keys()))
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown ticker(s): {', '.join(unknown)}")
    try:
        start_dt = datetime.strptime(request.backfill_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=400, detail="backfill_from must be YYYY-MM-DD")

    now         = datetime.now(timezone.utc)
    output_path = Path(request.output_path) if request.output_path else DATA_DIR / "ingested" / "tdwl_news_announcements.parquet"
    cursor_path = Path(request.cursor_path) if request.cursor_path else DATA_DIR / "ingested" / "tdwl_news_announcements.cursor.json"
    config      = MubasherRequestConfig(sid=request.sid, uid=request.uid)
    all_items   = []

    for item_type, rt, chunk in (
        ("news",         REQUEST_TYPE_NEWS,         NEWS_CHUNK_DAYS),
        ("announcement", REQUEST_TYPE_ANNOUNCEMENT, ANN_CHUNK_DAYS),
    ):
        all_items.extend(fetch_date_range(
            rt=rt, config=config, start_dt=start_dt, end_dt=now,
            item_type=item_type, allowed_tickers=tickers, chunk_days=chunk,
        ))

    before = len(load_ingested_items(output_path))
    merged = upsert_incremental_items(all_items, store_path=output_path)
    save_cursor(compute_latest_cursor(merged), cursor_path)
    return {"status":"ok", "records_added": max(len(merged)-before,0),
            "records_total": len(merged), "output_path": str(output_path), "cursor_path": str(cursor_path)}


# ── POST /ingestion/tdwl/today ────────────────────────────────────────────────
# @app.post("/ingestion/tdwl/today", response_model=IngestionRunResponse, tags=["Ingestion"])
async def run_tdwl_today(request):
    """Fetch today only (midnight to now). Call every 15 min during trading hours."""
    from src.ingestion.mubasher_tdwl import (
        MubasherRequestConfig, REQUEST_TYPE_ANNOUNCEMENT, REQUEST_TYPE_NEWS,
        compute_latest_cursor, fetch_date_range, filter_items_newer_than_cursor,
        load_cursor, save_cursor, upsert_incremental_items,
    )
    from src.services.news_signal import load_ingested_items

    tickers     = set(getattr(request,"tickers",None) or list(TICKERS.keys()))
    now         = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    output_path = DATA_DIR / "ingested" / "tdwl_news_announcements.parquet"
    cursor_path = DATA_DIR / "ingested" / "tdwl_news_announcements.cursor.json"
    config      = MubasherRequestConfig()
    cursor      = load_cursor(cursor_path)
    all_items   = []

    for item_type, rt in (("news", REQUEST_TYPE_NEWS), ("announcement", REQUEST_TYPE_ANNOUNCEMENT)):
        items = fetch_date_range(rt=rt, config=config, start_dt=today_start, end_dt=now,
                                 item_type=item_type, allowed_tickers=tickers, chunk_days=1)
        all_items.extend(filter_items_newer_than_cursor(items, cursor=cursor))

    before = len(load_ingested_items(output_path))
    merged = upsert_incremental_items(all_items, store_path=output_path)
    save_cursor(compute_latest_cursor(merged), cursor_path)
    return {"status":"ok", "records_added": max(len(merged)-before,0),
            "records_total": len(merged), "output_path": str(output_path), "cursor_path": str(cursor_path)}


# ── GET /news/headlines/{ticker} ──────────────────────────────────────────────
# @app.get("/news/headlines/{ticker}", tags=["News"])
async def get_headlines(ticker: str, limit: int = 20, source: str = "both"):
    """Return recent headlines with category classification tags."""
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")
    from src.services.news_signal import classify_category, load_ingested_items
    frame = load_ingested_items(DATA_DIR / "ingested" / "tdwl_news_announcements.parquet")
    if frame.empty:
        return []
    subset = frame[frame["ticker"] == ticker]
    if source != "both":
        subset = subset[subset["item_type"] == source]
    subset = subset.sort_values("published_at", ascending=False).head(limit)
    return [{"headline": r["headline"], "category": classify_category(r["headline"]),
             "item_type": r["item_type"], "published_at": str(r["published_at"])}
            for _, r in subset.iterrows()]


# ── GET /news/portfolio/signals ───────────────────────────────────────────────
# @app.get("/news/portfolio/signals", tags=["News"])
async def get_portfolio_signals(lookback_days: int = 1, model: str = "ensemble"):
    """Composite signals for all tickers — powers the dashboard portfolio strip."""
    if not predictor or not predictor.models_loaded():
        raise HTTPException(status_code=503, detail="Models not loaded.")
    from src.services.news_signal import (
        aggregate_directional_signal, build_prediction_rows,
        compute_macro_regime_score, load_ingested_items, select_recent_items,
    )
    as_of  = datetime.now(timezone.utc)
    frame  = load_ingested_items(DATA_DIR / "ingested" / "tdwl_news_announcements.parquet")
    results = {}
    for code in TICKERS:
        recent    = select_recent_items(frame, ticker=code, as_of=as_of, lookback_days=lookback_days, max_items=50)
        pred_rows = build_prediction_rows(frame=recent, predictor=predictor, ticker=code, model=model) if not recent.empty else []
        macro     = compute_macro_regime_score(recent) if not recent.empty else 0.0
        agg       = aggregate_directional_signal(pred_rows, as_of=as_of, macro_regime_score=macro)
        results[code] = {
            "company": TICKER_NAMES.get(code, code), "signal": agg.signal,
            "score": agg.score, "probability_up": agg.probability_up,
            "macro_dampener": agg.macro_dampener, "macro_regime": agg.macro_regime,
            "items_used": agg.item_count, "explanation": agg.explanation,
        }
    return results


# ── POST /predict/ticker/live  (REPLACES original) ────────────────────────────
# @app.post("/predict/ticker/live", response_model=LiveTickerSignalResponse, tags=["Prediction"])
async def predict_live_ticker_signal_v2(request):
    """
    Upgraded live signal endpoint.
    macro_regime_score is now auto-computed from actual ingested news.
    Returns macro_dampener, macro_regime, explanation, top_drivers.
    """
    if not predictor or not predictor.models_loaded():
        raise HTTPException(status_code=503, detail="Models not loaded.")
    if request.ticker not in TICKERS:
        raise HTTPException(status_code=400, detail=f"Unknown ticker '{request.ticker}'.")
    from src.services.news_signal import (
        aggregate_directional_signal, build_prediction_rows,
        compute_macro_regime_score, load_ingested_items, select_recent_items,
    )
    as_of = pd.to_datetime(request.as_of, utc=True).to_pydatetime() if request.as_of else datetime.now(timezone.utc)
    ingestion_path = Path(request.ingestion_path) if request.ingestion_path                      else DATA_DIR / "ingested" / "tdwl_news_announcements.parquet"
    frame  = load_ingested_items(ingestion_path)
    recent = select_recent_items(frame, ticker=request.ticker, as_of=as_of,
                                 lookback_days=max(request.lookback_days,1), max_items=max(request.max_items,1))
    pred_rows   = build_prediction_rows(frame=recent, predictor=predictor, ticker=request.ticker, model=request.model) if not recent.empty else []
    macro_score = compute_macro_regime_score(recent) if not recent.empty else request.macro_regime_score
    agg         = aggregate_directional_signal(pred_rows, as_of=as_of, macro_regime_score=macro_score)
    if not agg.ticker:
        agg.ticker = request.ticker
    return {
        "ticker": request.ticker, "company": TICKER_NAMES.get(request.ticker, request.ticker),
        "as_of": agg.as_of, "signal": agg.signal, "probability_up": agg.probability_up,
        "score": agg.score, "items_used": agg.item_count, "macro_dampener": agg.macro_dampener,
        "macro_regime": agg.macro_regime, "explanation": agg.explanation,
        "top_drivers": agg.top_drivers, "components": agg.components, "items": pred_rows,
    }
