import React, { useEffect, useMemo, useState } from 'react';
import apiService from '../services/api';
import PredictionPanel from './PredictionPanel';
import ModelComparison from './ModelComparison';
import { TICKERS as FALLBACK_TICKERS, TICKER_MAP } from '../config/constants';
import './Dashboard.css';

const Dashboard = () => {
  const [tickers, setTickers] = useState(FALLBACK_TICKERS);
  const [tickerMap, setTickerMap] = useState(TICKER_MAP);
  const [selectedTicker, setTicker] = useState('1120');
  const [modelMetrics, setMetrics] = useState(null);
  const [macroInfo, setMacroInfo] = useState({ available: false, latest_date: null, age_days: null });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => { fetchData(); }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [tickersRes, healthRes] = await Promise.all([
        apiService.getTickers(),
        apiService.health(),
      ]);

      const list = Object.entries(tickersRes.tickers || {}).map(([code, info]) => ({ code, ...info }));
      if (list.length) {
        setTickers(list);
        setTickerMap(Object.fromEntries(list.map((t) => [t.code, t])));
      }

      setMacroInfo(healthRes?.macro || { available: false, latest_date: null, age_days: null });

      const m = await apiService.getModelMetrics();
      setMetrics(m?.models || m?.metrics || m);
      setError(null);
    } catch {
      setError('Live backend data not available. Showing offline fallback metadata.');
    } finally {
      setLoading(false);
    }
  };

  const info = tickerMap[selectedTicker];
  const metricRows = useMemo(() => {
    if (!modelMetrics) return [];
    return Array.isArray(modelMetrics)
      ? modelMetrics
      : Object.entries(modelMetrics).map(([model, stats]) => ({ model, ...stats }));
  }, [modelMetrics]);

  const bestAcc = metricRows.reduce((acc, m) => Math.max(acc, m.accuracy || 0), 0);

  if (loading) return <div className="loading-screen"><div className="spinner" /><p>Loading Macro-News Intelligence Dashboard...</p></div>;

  return (
    <div className="dashboard">
      <header className="dash-header">
        <div>
          <p className="eyebrow">MSc Thesis Prototype · Real-time Macroeconomic News Intelligence</p>
          <h1>Stock Movement Prediction Studio</h1>
          <p className="subtext">Predicting directional movement from financial headlines, sector context and macro indicators.</p>
        </div>
        <div className="ticker-selector">
          <label htmlFor="ticker-select">Target Ticker</label>
          <select id="ticker-select" value={selectedTicker} onChange={(e) => setTicker(e.target.value)}>
            {tickers.map((t) => <option key={t.code} value={t.code}>{t.code} — {t.name}</option>)}
          </select>
        </div>
      </header>

      <main className="dash-body">
        {error && <div className="alert-error">⚠️ {error}</div>}

        <section className="kpi-grid">
          <article className="kpi-card"><h4>Coverage</h4><p>{tickers.length} Saudi stocks</p></article>
          <article className="kpi-card"><h4>Top Accuracy</h4><p>{(bestAcc * 100).toFixed(1)}%</p></article>
          <article className="kpi-card"><h4>Macro Snapshot</h4><p>{macroInfo.latest_date || 'Unavailable'}</p></article>
        </section>

        {info && <section className="card company-card"><h3>{info.name}</h3><p>{info.sector} · {info.description}</p></section>}

        <section className="card">
          <PredictionPanel
            ticker={selectedTicker}
            tickerInfo={info}
            macroLatestDate={macroInfo.latest_date}
            macroAgeDays={macroInfo.age_days}
          />
        </section>
        {modelMetrics && <section className="card"><h2>Model Benchmark Snapshot</h2><ModelComparison metrics={modelMetrics} /></section>}
      </main>

      <footer className="dash-footer">Designed for thesis demonstration and iterative validation against real macroeconomic signals.</footer>
    </div>
  );
};

export default Dashboard;
