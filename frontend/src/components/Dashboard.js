import React, { useEffect, useMemo, useState } from 'react';
import apiService from '../services/api';
import PredictionPanel from './PredictionPanel';
import ModelComparison from './ModelComparison';
import NewsSignalPanel from './NewsSignalPanel';
import { TICKERS as FALLBACK_TICKERS, TICKER_MAP } from '../config/constants';
import './Dashboard.css';

const SIGNAL_ICON = { BULLISH: '🟢', BEARISH: '🔴', NEUTRAL: '🟡', NO_DATA: '⚪' };

const Dashboard = () => {
  const [tickers,          setTickers]          = useState(FALLBACK_TICKERS);
  const [tickerMap,        setTickerMap]         = useState(TICKER_MAP);
  const [selectedTicker,   setTicker]            = useState('1120');
  const [modelMetrics,     setMetrics]           = useState(null);
  const [portfolioSignals, setPortfolioSignals]  = useState({});
  const [loading,          setLoading]           = useState(true);
  const [error,            setError]             = useState(null);
  const [activeTab,        setActiveTab]         = useState('news');

  useEffect(() => { fetchData(); }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const res  = await apiService.getTickers();
      const list = Object.entries(res.tickers || {}).map(([code, info]) => ({ code, ...info }));
      if (list.length) {
        setTickers(list);
        setTickerMap(Object.fromEntries(list.map((t) => [t.code, t])));
      }
      const m = await apiService.getModelMetrics();
      setMetrics(m?.models || m?.metrics || m);
      try {
        const sigs = await apiService.getPortfolioSignals(1);
        setPortfolioSignals(sigs || {});
      } catch { /* non-fatal */ }
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

  const bestAcc    = metricRows.reduce((acc, m) => Math.max(acc, m.accuracy || 0), 0);
  const bullishCnt = Object.values(portfolioSignals).filter((s) => s.signal === 'BULLISH').length;
  const bearishCnt = Object.values(portfolioSignals).filter((s) => s.signal === 'BEARISH').length;

  if (loading) return (
    <div className="loading-screen">
      <div className="spinner" />
      <p>Loading Macro-News Intelligence Dashboard…</p>
    </div>
  );

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
            {tickers.map((t) => (
              <option key={t.code} value={t.code}>
                {SIGNAL_ICON[portfolioSignals[t.code]?.signal] || '⚪'} {t.code} — {t.name}
              </option>
            ))}
          </select>
        </div>
      </header>

      <main className="dash-body">
        {error && <div className="alert-error">⚠️ {error}</div>}

        <section className="kpi-grid">
          <article className="kpi-card"><h4>Coverage</h4><p>{tickers.length} Saudi stocks</p></article>
          <article className="kpi-card"><h4>Top Accuracy</h4><p>{(bestAcc * 100).toFixed(1)}%</p></article>
          <article className="kpi-card">
            <h4>Today's Signals</h4>
            <p>🟢 {bullishCnt} Bullish · 🔴 {bearishCnt} Bearish</p>
          </article>
        </section>

        {Object.keys(portfolioSignals).length > 0 && (
          <section className="card portfolio-strip">
            <h3 className="strip-title">Portfolio Overview — Live Signals</h3>
            <div className="strip-grid">
              {Object.entries(portfolioSignals).map(([code, s]) => (
                <div
                  key={code}
                  className={`strip-item ${(s.signal || 'neutral').toLowerCase()}`}
                  onClick={() => setTicker(code)}
                  title={s.explanation || ''}
                  style={{ cursor: 'pointer' }}
                >
                  <span className="strip-ticker">{code}</span>
                  <span className="strip-signal">{SIGNAL_ICON[s.signal] || '⚪'} {s.signal}</span>
                  <span className="strip-score">{s.score > 0 ? '+' : ''}{s.score?.toFixed(2)}</span>
                  {s.macro_dampener > 0.05 && <span className="strip-warn" title="Macro dampener active">⚠️</span>}
                </div>
              ))}
            </div>
          </section>
        )}

        {info && (
          <section className="card company-card">
            <h3>{info.name}</h3>
            <p>{info.sector} · {info.description}</p>
          </section>
        )}

        <div className="tab-nav">
          <button className={`tab-btn ${activeTab === 'news'    ? 'active' : ''}`} onClick={() => setActiveTab('news')}>📰 Live News Signal</button>
          <button className={`tab-btn ${activeTab === 'predict' ? 'active' : ''}`} onClick={() => setActiveTab('predict')}>🔮 Manual Inference</button>
        </div>

        <section className="card">
          {activeTab === 'news'
            ? <NewsSignalPanel ticker={selectedTicker} tickerInfo={info} lookbackDays={1} />
            : <PredictionPanel ticker={selectedTicker} tickerInfo={info} />
          }
        </section>

        {modelMetrics && (
          <section className="card">
            <h2>Model Benchmark Snapshot</h2>
            <ModelComparison metrics={modelMetrics} />
          </section>
        )}
      </main>

      <footer className="dash-footer">
        Designed for thesis demonstration and iterative validation against real macroeconomic signals.
      </footer>
    </div>
  );
};

export default Dashboard;
