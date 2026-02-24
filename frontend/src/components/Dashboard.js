// src/components/Dashboard.js
import React, { useState, useEffect } from 'react';
import apiService from '../services/api';
import PredictionPanel from './PredictionPanel';
import ModelComparison from './ModelComparison';
import { TICKERS as FALLBACK_TICKERS, TICKER_MAP } from '../config/constants';
import './Dashboard.css';

const Dashboard = () => {
  const [tickers, setTickers]         = useState(FALLBACK_TICKERS);
  const [tickerMap, setTickerMap]     = useState(TICKER_MAP);
  const [selectedTicker, setTicker]   = useState('1120');
  const [modelMetrics, setMetrics]    = useState(null);
  const [loading, setLoading]         = useState(true);
  const [error, setError]             = useState(null);

  useEffect(() => { fetchData(); }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const res = await apiService.getTickers();
      const list = Object.entries(res.tickers || {}).map(([code, info]) => ({
        code, name: info.name, sector: info.sector, description: info.description,
      }));
      if (list.length > 0) {
        setTickers(list);
        setTickerMap(Object.fromEntries(list.map(t => [t.code, t])));
      }
      try {
  const m = await apiService.getModelMetrics();
  // Normalize: could be { models: [...] } or { metrics: [...] } or direct array/object
  const metricsData = m?.models || m?.metrics || m;
  setMetrics(metricsData);
} catch {}
      setError(null);
    } catch {
      setError('Cannot connect to backend — using local ticker list.');
    } finally {
      setLoading(false);
    }
  };

  if (loading) return (
    <div className="loading-screen">
      <div className="spinner" />
      <p>Loading Tadawul Prediction System...</p>
    </div>
  );

  const info = tickerMap[selectedTicker];

  return (
    <div className="dashboard">

      {/* ── HEADER with ticker selector ── */}
      <header className="dash-header">
        <div>
          <h1>📈 Tadawul Stock Movement Prediction</h1>
          <p>News-Driven Machine Learning System — M I M Safras (248270E)</p>
        </div>

        <div className="ticker-selector">
          <label htmlFor="ticker-select" style={{ color: 'white', fontWeight: 600 }}>
            Stock:
          </label>
          <select
            id="ticker-select"
            value={selectedTicker}
            onChange={(e) => setTicker(e.target.value)}
          >
            {tickers.map((t) => (
              <option key={t.code} value={t.code}>
                {t.code} — {t.name}
              </option>
            ))}
          </select>
          <span style={{ color: 'rgba(255,255,255,0.75)', fontSize: '0.85rem' }}>
            {tickers.length} tickers
          </span>
        </div>
      </header>

      {/* ── BODY ── */}
      <main className="dash-body">

        {error && (
          <div className="alert-error">⚠️ {error}</div>
        )}

        {/* Ticker info bar */}
        {info && (
          <div className="card" style={{ padding: '1rem 2rem' }}>
            <strong style={{ fontSize: '1.1rem', color: '#2d3748' }}>{info.name}</strong>
            <span style={{ margin: '0 0.75rem', color: '#718096' }}>·</span>
            <span style={{ color: '#667eea', fontWeight: 600 }}>{info.sector}</span>
            <span style={{ margin: '0 0.75rem', color: '#718096' }}>·</span>
            <span style={{ color: '#4a5568' }}>{info.description}</span>
          </div>
        )}

        {/* Prediction card */}
        <div className="card">
          <PredictionPanel ticker={selectedTicker} tickerInfo={info} />
        </div>

        {/* Model metrics card */}
        {modelMetrics && (
          <div className="card">
            <h2>📊 Model Performance</h2>
            <ModelComparison metrics={modelMetrics} />
          </div>
        )}

      </main>

      {/* ── FOOTER ── */}
      <footer className="dash-footer">
        Tadawul Stock Movement Prediction · M I M Safras (248270E) · MSc Project 2026
      </footer>

    </div>
  );
};

export default Dashboard;
