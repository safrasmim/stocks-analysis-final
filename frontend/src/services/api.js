import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

api.interceptors.response.use(
  (res) => res.data,
  (err) => {
    console.error('API Error:', err.response?.data || err.message);
    throw err;
  }
);

const apiService = {
  // ── Existing endpoints (unchanged) ────────────────────────────────────────
  health:          ()                                                               => api.get('/health'),
  getTickers:      ()                                                               => api.get('/tickers'),
  getModels:       ()                                                               => api.get('/models'),
  getModelMetrics: ()                                                               => api.get('/models/metrics'),
  predict:         (ticker, news_texts, model_type = 'ensemble', event_date = null) =>
                     api.post('/predict', { ticker, news_texts, model_type, event_date }),

  // ── Ingestion ─────────────────────────────────────────────────────────────
  /** Fetch today's news + announcements into the store (call every 15 min) */
  fetchToday: (ticker = null) =>
    api.post('/ingestion/tdwl/today', ticker ? { tickers: [ticker] } : {}),

  /** One-time full history backfill from a start date */
  backfill: (backfill_from = '2025-01-01', tickers = null) =>
    api.post('/ingestion/tdwl/backfill', { backfill_from, ...(tickers ? { tickers } : {}) }),

  /** Incremental cursor-based update */
  ingestIncremental: (tickers = null) =>
    api.post('/ingestion/tdwl/run', tickers ? { tickers } : {}),

  // ── Live signals ──────────────────────────────────────────────────────────
  /** Composite signal for one ticker with category weighting + macro dampener */
  getLiveSignal: (ticker, lookback_days = 1, model = 'ensemble') =>
    api.post('/predict/ticker/live', { ticker, lookback_days, model }),

  /** Signals for ALL tickers — used by portfolio strip in Dashboard */
  getPortfolioSignals: (lookback_days = 1, model = 'ensemble') =>
    api.get('/news/portfolio/signals', { params: { lookback_days, model } }),

  /** Recent headlines with category tags */
  getHeadlines: (ticker, limit = 20, source = 'both') =>
    api.get(`/news/headlines/${ticker}`, { params: { limit, source } }),

  // ── Broadcast (existing) ──────────────────────────────────────────────────
  broadcast: (headline) =>
    api.post('/predict/broadcast', { headline }),
};

export default apiService;
