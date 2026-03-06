import React, { useState, useEffect, useCallback } from 'react';
import apiService from '../services/api';
import './NewsSignalPanel.css';

const SIGNAL_CONFIG = {
  BULLISH: { icon: '🟢', color: '#065f46', bg: '#d1fae5', label: 'Bullish' },
  BEARISH: { icon: '🔴', color: '#991b1b', bg: '#fee2e2', label: 'Bearish' },
  NEUTRAL: { icon: '🟡', color: '#92400e', bg: '#fef3c7', label: 'Neutral' },
  NO_DATA: { icon: '⚪', color: '#6b7280', bg: '#f3f4f6', label: 'No Data' },
};

const CATEGORY_BADGE = {
  geopolitical:   { bg: '#fee2e2', color: '#991b1b', label: 'Geopolitical' },
  macro_economic: { bg: '#fef3c7', color: '#92400e', label: 'Macro'        },
  sector_wide:    { bg: '#dbeafe', color: '#1e40af', label: 'Sector'       },
  company:        { bg: '#d1fae5', color: '#065f46', label: 'Company'      },
  general:        { bg: '#f3f4f6', color: '#4b5563', label: 'General'      },
};

const NewsSignalPanel = ({ ticker, tickerInfo, lookbackDays = 1 }) => {
  const [signal,      setSignal]      = useState(null);
  const [headlines,   setHeadlines]   = useState([]);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState(null);
  const [lastRefresh, setLastRefresh] = useState(null);

  const refresh = useCallback(async () => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    try {
      try { await apiService.fetchToday(ticker); } catch { /* non-fatal */ }
      const sig = await apiService.getLiveSignal(ticker, lookbackDays);
      setSignal(sig);
      const hdl = await apiService.getHeadlines(ticker, 15);
      setHeadlines(Array.isArray(hdl) ? hdl : []);
      setLastRefresh(new Date());
    } catch (e) {
      setError(e?.response?.data?.detail || 'Signal fetch failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  }, [ticker, lookbackDays]);

  useEffect(() => { refresh(); }, [refresh]);

  const cfg = SIGNAL_CONFIG[signal?.signal] ?? SIGNAL_CONFIG.NO_DATA;

  return (
    <div className="nsp-wrap">
      <div className="nsp-header">
        <div>
          <h2 className="nsp-title">Live News Signal</h2>
          <p className="nsp-sub">{tickerInfo?.name || ticker} · last {lookbackDays}d lookback</p>
        </div>
        <button className="nsp-refresh-btn" onClick={refresh} disabled={loading}>
          {loading ? 'Refreshing…' : '⟳ Refresh'}
        </button>
      </div>

      {error && <div className="nsp-error">⚠️ {error}</div>}

      {signal && (
        <div className="nsp-signal-card" style={{ background: cfg.bg, borderColor: cfg.color }}>
          <div className="nsp-signal-main">
            <span className="nsp-signal-icon">{cfg.icon}</span>
            <span className="nsp-signal-label" style={{ color: cfg.color }}>{cfg.label}</span>
            <span className="nsp-signal-score">
              score: {signal.score > 0 ? '+' : ''}{signal.score?.toFixed(3)}
            </span>
            <span className="nsp-signal-conf">
              {(Math.min(Math.abs(signal.score ?? 0) * 2, 1) * 100).toFixed(0)}% confidence
            </span>
          </div>

          {signal.macro_dampener > 0.05 && (
            <div className="nsp-dampener-warn">
              ⚠️ Macro/geopolitical pressure active — positive signals dampened by{' '}
              {(signal.macro_dampener * 100).toFixed(0)}%
              {' '}(macro regime: {signal.macro_regime > 0 ? '+' : ''}{signal.macro_regime?.toFixed(2)})
            </div>
          )}

          {signal.explanation && <p className="nsp-explanation">{signal.explanation}</p>}

          <div className="nsp-metrics">
            <div className="nsp-metric">
              <span>Prob. Up</span>
              <strong>{((signal.probability_up ?? 0.5) * 100).toFixed(1)}%</strong>
            </div>
            <div className="nsp-metric">
              <span>Items Used</span>
              <strong>{signal.items_used ?? 0}</strong>
            </div>
            <div className="nsp-metric">
              <span>As Of</span>
              <strong>{signal.as_of}</strong>
            </div>
          </div>
        </div>
      )}

      {signal?.top_drivers?.length > 0 && (
        <div className="nsp-section">
          <h3 className="nsp-section-title">Key Drivers</h3>
          {signal.top_drivers.map((d, i) => {
            const cat = CATEGORY_BADGE[d.category] ?? CATEGORY_BADGE.general;
            return (
              <div key={i} className="nsp-driver-row">
                <span className="nsp-cat-badge" style={{ background: cat.bg, color: cat.color }}>
                  {cat.label}
                </span>
                <span className={`nsp-dir ${d.direction === 'UP' ? 'up' : d.direction === 'DOWN' ? 'down' : 'neutral'}`}>
                  {d.direction === 'UP' ? '▲' : d.direction === 'DOWN' ? '▼' : '—'}
                </span>
                <span className="nsp-driver-text">{d.headline}</span>
                <span className="nsp-weight">w={d.weight?.toFixed(2)}</span>
              </div>
            );
          })}
        </div>
      )}

      {headlines.length > 0 && (
        <div className="nsp-section">
          <h3 className="nsp-section-title">Recent Headlines</h3>
          {headlines.map((h, i) => {
            const cat = CATEGORY_BADGE[h.category] ?? CATEGORY_BADGE.general;
            return (
              <div key={i} className="nsp-headline-row">
                <span className="nsp-cat-badge" style={{ background: cat.bg, color: cat.color }}>
                  {cat.label}
                </span>
                <span className="nsp-headline-type">{h.item_type}</span>
                <span className="nsp-headline-text">{h.headline}</span>
                <span className="nsp-headline-dt">
                  {new Date(h.published_at).toLocaleDateString('en-SA')}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {lastRefresh && (
        <p className="nsp-last-refresh">Last updated: {lastRefresh.toLocaleTimeString()}</p>
      )}
    </div>
  );
};

export default NewsSignalPanel;
