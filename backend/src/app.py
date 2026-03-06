"""
app.py
FastAPI application — Tadawul Stock News Sentiment Predictor
MSc Thesis Project — February 2026

Endpoints:
  GET  /health                → system status + model readiness
  GET  /tickers               → all configured tickers from config.py
  GET  /tickers/{ticker}      → single ticker info
  GET  /models/metrics        → full evaluation metrics (precision/recall/f1/auc)
  POST /predict               → UP/DOWN for headlines for one ticker
  POST /predict/batch         → same headline for multiple tickers
  POST /predict/broadcast     → macro headline → all tickers
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)

import csv
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import (
    MODEL_DIR, MODELS_DIR, DATA_DIR,
    TICKERS, TICKER_NAMES,
    API_TITLE, API_VERSION, API_DESCRIPTION, CORS_ORIGINS,
    REQUIRE_EVENT_DATE,
)
from .predictor import Predictor
from .ingestion.mubasher_tdwl import (
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
from .services.news_signal import (
    aggregate_directional_signal,
    build_prediction_rows,
    load_ingested_items,
    select_recent_items,
)

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = API_TITLE,
    description = API_DESCRIPTION,
    version     = API_VERSION,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = CORS_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Load predictor once at startup ─────────────────────────────────────────────
predictor: Optional[Predictor] = None

@app.on_event("startup")
async def startup():
    global predictor
    try:
        predictor = Predictor(model_dir=MODELS_DIR)
        predictor.load()
        logger.info("✅ Models loaded successfully.")
    except Exception as e:
        logger.error("❌ Failed to load models: %s", e)
        logger.error("   Run: python scripts/train_all_models.py")
        predictor = None

# ── Company keyword map for mismatch warnings ──────────────────────────────────
TICKER_KEYWORDS: Dict[str, List[str]] = {
    "1120": ["al rajhi", "alrajhi", "rajhi"],
    "1150": ["alinma"],
    "1050": ["ncb", "national commercial"],
    "1060": ["riyad bank"],
    "1080": ["arab national", "anb"],
    "1090": ["saudi investment bank", "saib"],
    "1100": ["banque saudi fransi", "bsf"],
    "1110": ["al jazira", "aljazira"],
    "1211": ["maaden", "ma'aden"],
    "2010": ["sabic"],
    "2222": ["aramco", "saudi aramco"],
    "2350": ["saudi kayan", "kayan"],
    "4110": ["tawuniya"],
    "4190": ["jarir"],
    "4210": ["almarai"],
    "4230": ["extra"],
    "4325": ["masar"],
    "4050": ["savola"],
    "7010": ["stc", "saudi telecom"],
    "7020": ["mobily", "etihad etisalat"],
    "7030": ["zain"],
}


def _mismatch_warning(headline: str, ticker: str) -> Optional[str]:
    h_lower      = headline.lower()
    selected_kws = TICKER_KEYWORDS.get(ticker, [])
    for other_ticker, keywords in TICKER_KEYWORDS.items():
        if other_ticker == ticker:
            continue
        for kw in keywords:
            if kw in h_lower and not any(k in h_lower for k in selected_kws):
                other_name    = TICKER_NAMES.get(other_ticker, other_ticker)
                selected_name = TICKER_NAMES.get(ticker, ticker)
                return (
                    f"⚠️ Headline mentions '{other_name}' "
                    f"but selected stock is {selected_name} ({ticker}). "
                    f"Consider switching to ticker {other_ticker}."
                )
    return None


def _signal_from_probability(prob_up: float) -> str:
    """Convert probability to investor-facing signal."""
    if prob_up >= 0.55:
        return "BULLISH"
    if prob_up <= 0.45:
        return "BEARISH"
    return "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    """
    Accepts BOTH field naming conventions:
      headlines  = internal API standard
      news_texts = legacy frontend field name
      model / model_type = both accepted
    """
    ticker:     str
    headlines:  Optional[List[str]] = None
    news_texts: Optional[List[str]] = None
    model:       Optional[str]       = "ensemble"
    model_type:  Optional[str]       = None
    event_date:  Optional[str]       = None
    event_dates: Optional[List[str]] = None

    def get_texts(self) -> List[str]:
        return self.headlines or self.news_texts or []

    def get_model(self) -> str:
        return self.model_type or self.model or "ensemble"

    def get_event_dates(self, n_texts: int) -> List[str]:
        if self.event_dates and len(self.event_dates) == n_texts:
            return self.event_dates
        if self.event_date:
            return [self.event_date] * n_texts
        return [datetime.utcnow().strftime("%Y-%m-%d")] * n_texts

    class Config:
        json_schema_extra = {"example": {
            "ticker": "2010",
            "news_texts": ["SABIC reports record quarterly profit"],
            "model_type": "ensemble",
        }}


class HeadlineResult(BaseModel):
    headline:       str
    prediction:     str
    signal:         str
    confidence:     float
    probability_up: float
    warning:        Optional[str] = None


class PredictResponse(BaseModel):
    ticker:     str
    company:    str
    results:    List[HeadlineResult]
    model_used: str = "ensemble"


class BatchPredictRequest(BaseModel):
    tickers:  List[str]
    headline: str


class BatchPredictItem(BaseModel):
    ticker:         str
    company:        str
    prediction:     str
    signal:         str
    confidence:     float
    probability_up: float


class BatchPredictResponse(BaseModel):
    headline: str
    results:  List[BatchPredictItem]


class BroadcastRequest(BaseModel):
    headline: str
    class Config:
        json_schema_extra = {"example": {
            "headline": "US Federal Reserve cuts rates by 50 basis points"
        }}


class BroadcastStockResult(BaseModel):
    ticker:         str
    name:           str
    sector:         str
    direction:      str
    signal:         str
    confidence:     float
    probability_up: float


class BroadcastResponse(BaseModel):
    headline:     str
    total_stocks: int
    by_sector:    Dict[str, List[BroadcastStockResult]]
    most_bullish: List[BroadcastStockResult]
    most_bearish: List[BroadcastStockResult]


class IngestionRunRequest(BaseModel):
    tickers: Optional[List[str]] = None
    sid: str = "sid"
    uid: str = "123"
    language: str = "EN"
    output_path: Optional[str] = None
    cursor_path: Optional[str] = None
    extra_params: Dict[str, str] = Field(default_factory=dict)


class IngestionRunResponse(BaseModel):
    status: str
    records_added: int
    records_total: int
    output_path: str
    cursor_path: str


class LiveTickerSignalRequest(BaseModel):
    ticker: str
    model: str = "ensemble"
    as_of: Optional[str] = None
    lookback_days: int = 3
    max_items: int = 50
    macro_regime_score: float = 0.0
    ingestion_path: Optional[str] = None



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


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── GET /health ────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    """System status and model readiness."""
    models_ready = predictor.models_loaded() if predictor else False
    macro = predictor.macro_status() if predictor else {"available": False, "latest_date": None, "age_days": None}
    return {
        "status":        "ok" if models_ready else "degraded",
        "models_ready":  models_ready,
        "tickers_count": len(TICKERS),
        "macro":         macro,
        "message":       "All systems operational." if models_ready
                         else "Models not loaded — run train_all_models.py",
    }


# ── GET /tickers ───────────────────────────────────────────────────────────────
@app.get("/tickers", tags=["Tickers"])
async def get_tickers():
    """All configured tickers — auto-updates when you add new ones to config.py."""
    return {"tickers": TICKERS, "count": len(TICKERS)}


# ── GET /tickers/{ticker} ──────────────────────────────────────────────────────
@app.get("/tickers/{ticker}", tags=["Tickers"])
async def get_ticker(ticker: str):
    """Info for a single ticker."""
    if ticker not in TICKERS:
        raise HTTPException(
            status_code = 404,
            detail      = f"Ticker '{ticker}' not found. Valid: {sorted(TICKERS.keys())}",
        )
    return {"ticker": ticker, **TICKERS[ticker]}


# ── GET /models/metrics ────────────────────────────────────────────────────────
@app.get("/models/metrics", tags=["Models"])
async def get_metrics():
    """
    Return FULL evaluation metrics: accuracy, precision, recall, f1_score, roc_auc.
    Priority: JSON file → CSV file.
    """
    json_path = DATA_DIR / "evaluation" / "evaluation_metrics.json"
    csv_path  = DATA_DIR / "evaluation" / "evaluation_metrics.csv"

    # ── 1. Try JSON (richest — includes all fields + confusion matrix) ─────────
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            # Strip confusion_matrix from API response (too large for table)
            clean = [{k: v for k, v in m.items() if k != "confusion_matrix"} for m in data]
            return {"available": True, "source": "json", "metrics": clean, "count": len(clean)}
        except Exception as e:
            logger.warning("Could not read metrics JSON: %s", e)

    # ── 2. Try CSV ─────────────────────────────────────────────────────────────
    if csv_path.exists():
        try:
            metrics = []
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    cleaned = {"model": row.get("model", row.get("Model", ""))}
                    for k, v in row.items():
                        if k.lower().strip() == "model":
                            continue
                        try:
                            cleaned[k.lower().strip()] = round(float(v), 4)
                        except (ValueError, TypeError):
                            cleaned[k.lower().strip()] = v
                    metrics.append(cleaned)
            if metrics:
                return {"available": True, "source": "csv", "metrics": metrics, "count": len(metrics)}
        except Exception as e:
            logger.warning("Could not read metrics CSV: %s", e)

    # ── 3. No fallback in production/research mode ───────────────────────────
    return {
        "available": False,
        "source":    "none",
        "message":   "Evaluation metrics are unavailable. Run: python scripts/evaluate_all_models.py",
        "metrics":   [],
        "count":     0,
    }


# ── POST /predict ──────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict UP/DOWN for each headline.
    Accepts both {headlines}/{news_texts} and {model}/{model_type}.
    """
    if not predictor or not predictor.models_loaded():
        raise HTTPException(status_code=503, detail="Models not loaded. Run train_all_models.py first.")

    if request.ticker not in TICKERS:
        raise HTTPException(status_code=400, detail=f"Unknown ticker: {request.ticker}")

    texts = request.get_texts()
    if not texts:
        raise HTTPException(status_code=400, detail="Provide headlines or news_texts (non-empty list).")
    if REQUIRE_EVENT_DATE and not (request.event_date or request.event_dates):
        raise HTTPException(status_code=400, detail="event_date (or event_dates) is required for macro-aware inference.")

    model_name = request.get_model()
    event_dates = request.get_event_dates(len(texts))
    results: List[HeadlineResult] = []

    for headline, event_date in zip(texts, event_dates):
        if not headline.strip():
            continue
        try:
            pred = predictor.predict(
                texts=[headline],
                ticker=request.ticker,
                model=model_name,
                event_dates=[event_date],
            )
            prob_up    = float(pred["probabilities_up"][0])
            direction  = "UP" if pred["predictions"][0] == 1 else "DOWN"
            confidence = round((prob_up if direction == "UP" else 1 - prob_up) * 100, 1)
            model_warning = pred.get("warning")
            mismatch_warning = _mismatch_warning(headline, request.ticker)
            combined_warning = " | ".join(w for w in [model_warning, mismatch_warning] if w)
            results.append(HeadlineResult(
                headline       = headline,
                prediction     = direction,
                signal         = _signal_from_probability(prob_up),
                confidence     = confidence,
                probability_up = round(prob_up, 4),
                warning        = combined_warning or None,
            ))
        except Exception as e:
            logger.error("Prediction error: %s", e)
            results.append(HeadlineResult(
                headline       = headline,
                prediction     = "UNKNOWN",
                signal         = "NEUTRAL",
                confidence     = 0.0,
                probability_up = 0.0,
                warning        = f"Prediction error: {str(e)}",
            ))

    return PredictResponse(
        ticker     = request.ticker,
        company    = TICKER_NAMES.get(request.ticker, request.ticker),
        results    = results,
        model_used = model_name,
    )


# ── POST /predict/batch ────────────────────────────────────────────────────────
@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """Same headline predicted for multiple tickers — useful for impact comparison."""
    if not predictor or not predictor.models_loaded():
        raise HTTPException(status_code=503, detail="Models not loaded.")
    if not request.headline.strip():
        raise HTTPException(status_code=400, detail="headline cannot be empty.")

    results: List[BatchPredictItem] = []
    for ticker in request.tickers:
        if ticker not in TICKERS:
            continue
        try:
            pred       = predictor.predict(texts=[request.headline], ticker=ticker, model="ensemble", event_dates=[datetime.utcnow().strftime("%Y-%m-%d")])
            prob_up    = float(pred["probabilities_up"][0])
            direction  = "UP" if pred["predictions"][0] == 1 else "DOWN"
            confidence = round((prob_up if direction == "UP" else 1 - prob_up) * 100, 1)
            results.append(BatchPredictItem(
                ticker         = ticker,
                company        = TICKER_NAMES.get(ticker, ticker),
                prediction     = direction,
                signal         = _signal_from_probability(prob_up),
                confidence     = confidence,
                probability_up = round(prob_up, 4),
            ))
        except Exception as e:
            logger.error("Batch error for %s: %s", ticker, e)

    return BatchPredictResponse(headline=request.headline, results=results)


# ── POST /predict/broadcast ────────────────────────────────────────────────────
@app.post("/predict/broadcast", response_model=BroadcastResponse, tags=["Prediction"])
async def predict_broadcast(request: BroadcastRequest):
    """Macro headline → predictions for ALL configured tickers."""
    if not predictor or not predictor.models_loaded():
        raise HTTPException(status_code=503, detail="Models not loaded.")
    if not request.headline.strip():
        raise HTTPException(status_code=400, detail="headline cannot be empty.")

    all_results: List[BroadcastStockResult] = []
    for ticker, info in TICKERS.items():
        try:
            pred       = predictor.predict(texts=[request.headline], ticker=ticker, model="ensemble", event_dates=[datetime.utcnow().strftime("%Y-%m-%d")])
            prob_up    = float(pred["probabilities_up"][0])
            direction  = "UP" if pred["predictions"][0] == 1 else "DOWN"
            confidence = round((prob_up if direction == "UP" else 1 - prob_up) * 100, 1)
            all_results.append(BroadcastStockResult(
                ticker         = ticker,
                name           = info["name"],
                sector         = info["sector"],
                direction      = direction,
                signal         = _signal_from_probability(prob_up),
                confidence     = confidence,
                probability_up = round(prob_up, 4),
            ))
        except Exception as e:
            logger.error("Broadcast error for %s: %s", ticker, e)

    all_results.sort(key=lambda x: x.probability_up)

    by_sector: Dict[str, List[BroadcastStockResult]] = {}
    for r in all_results:
        by_sector.setdefault(r.sector, []).append(r)

    return BroadcastResponse(
        headline     = request.headline,
        total_stocks = len(all_results),
        by_sector    = by_sector,
        most_bearish = all_results[:5],
        most_bullish = list(reversed(all_results[-5:])),
    )


@app.post("/ingestion/tdwl/run", response_model=IngestionRunResponse, tags=["Ingestion"])
async def run_tdwl_ingestion(request: IngestionRunRequest):
    """Run TDWL ingestion in-process and persist only records newer than cursor."""
    tickers = set(request.tickers or list(TICKERS.keys()))
    unknown = sorted(tickers - set(TICKERS.keys()))
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown ticker(s): {', '.join(unknown)}")

    output_path = Path(request.output_path) if request.output_path else (DATA_DIR / "ingested" / "tdwl_news_announcements.parquet")
    cursor_path = Path(request.cursor_path) if request.cursor_path else (DATA_DIR / "ingested" / "tdwl_news_announcements.cursor.json")

    config = MubasherRequestConfig(sid=request.sid, uid=request.uid, language=request.language)
    cursor = load_cursor(cursor_path)

    new_items = []
    for item_type, rt in (("announcement", REQUEST_TYPE_ANNOUNCEMENT), ("news", REQUEST_TYPE_NEWS)):
        url = build_mix2_url(rt=rt, config=config, extra_params=request.extra_params)
        payload = fetch_mix2_json(url)
        parsed = parse_mix2_items(payload, item_type=item_type, allowed_tickers=tickers)
        new_items.extend(filter_items_newer_than_cursor(parsed, cursor=cursor))

    before = len(load_ingested_items(output_path))
    merged = upsert_incremental_items(new_items, store_path=output_path)
    save_cursor(compute_latest_cursor(merged), cursor_path)

    return IngestionRunResponse(
        status="ok",
        records_added=max(len(merged) - before, 0),
        records_total=len(merged),
        output_path=str(output_path),
        cursor_path=str(cursor_path),
    )


@app.post("/predict/ticker/live", response_model=LiveTickerSignalResponse, tags=["Prediction"])
async def predict_live_ticker_signal(request: LiveTickerSignalRequest):
    """Aggregate latest ticker news+announcements into a single thesis-aligned signal."""
    if not predictor or not predictor.models_loaded():
        raise HTTPException(status_code=503, detail="Models not loaded.")
    if request.ticker not in TICKERS:
        raise HTTPException(status_code=400, detail=f"Unknown ticker '{request.ticker}'.")

    as_of = pd.to_datetime(request.as_of, utc=True).to_pydatetime() if request.as_of else datetime.now(timezone.utc)
    ingestion_path = Path(request.ingestion_path) if request.ingestion_path else (DATA_DIR / "ingested" / "tdwl_news_announcements.parquet")

    frame = load_ingested_items(ingestion_path)
    recent = select_recent_items(
        frame,
        ticker=request.ticker,
        as_of=as_of,
        lookback_days=max(request.lookback_days, 1),
        max_items=max(request.max_items, 1),
    )
    prediction_rows = build_prediction_rows(
        frame=recent,
        predictor=predictor,
        ticker=request.ticker,
        model=request.model,
    ) if not recent.empty else []

    aggregate = aggregate_directional_signal(
        prediction_rows,
        as_of=as_of,
        macro_regime_score=request.macro_regime_score,
    )
    if not aggregate.ticker:
        aggregate.ticker = request.ticker

    return LiveTickerSignalResponse(
        ticker=request.ticker,
        company=TICKER_NAMES.get(request.ticker, request.ticker),
        as_of=aggregate.as_of,
        signal=aggregate.signal,
        probability_up=aggregate.probability_up,
        score=aggregate.score,
        items_used=aggregate.item_count,
        components=aggregate.components,
        items=prediction_rows,
    )


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
