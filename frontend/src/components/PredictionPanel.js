// src/components/PredictionPanel.js
import React, { useState } from 'react';
import apiService from '../services/api';
import './PredictionPanel.css';

const PredictionPanel = ({ ticker, tickerInfo }) => {
  const [newsInput, setNewsInput]     = useState('');
  const [modelType, setModelType]     = useState('ensemble');
  const [predictions, setPredictions] = useState([]);
  const [modelUsed, setModelUsed]     = useState('');
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);

  const handlePredict = async () => {
    const lines = newsInput.split('\n').map(l => l.trim()).filter(Boolean);
    if (!lines.length) { setError('Enter at least one headline.'); return; }
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.predict(ticker, lines, modelType);
      // ✅ API returns { results: [...], model_used: "..." }
      setPredictions(res.results || res.predictions || []);
      setModelUsed(res.model_used || modelType);
    } catch {
      setError('Prediction failed. Is the backend running on port 8000?');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => { setNewsInput(''); setPredictions([]); setError(null); setModelUsed(''); };

  return (
    <div className="pred-panel">
      <h2>🔮 Make Prediction</h2>

      <div className="controls">
        <div className="ctrl-group">
          <label>Model</label>
          <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
            <option value="ensemble">Ensemble (Best)</option>
            <option value="random_forest">Random Forest</option>
            <option value="xgboost">XGBoost</option>
            <option value="lstm">LSTM</option>
          </select>
        </div>

        <div className="ctrl-group full">
          <label>News Headlines — one per line</label>
          <textarea
            rows={5}
            value={newsInput}
            onChange={(e) => setNewsInput(e.target.value)}
            placeholder={
              tickerInfo
                ? `${tickerInfo.name} reports record quarterly profit\nSaudi Aramco raises dividend by 10%`
                : 'Enter news headlines here...'
            }
          />
        </div>
      </div>

      <div className="btn-row">
        <button className="btn-primary" onClick={handlePredict} disabled={loading}>
          {loading ? 'Predicting…' : '🚀 Predict Movement'}
        </button>
        <button className="btn-secondary" onClick={handleClear}>🗑 Clear</button>
      </div>

      {error && <div className="alert-error">⚠️ {error}</div>}

      {predictions.length > 0 && (
        <div className="results">
          <h3>
            Results for {tickerInfo?.name || ticker}
            <span style={{ fontSize: '0.85rem', color: '#718096', marginLeft: '0.75rem' }}>
              via {modelUsed.replace('_', ' ')}
            </span>
          </h3>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Headline</th>
                <th>Prediction</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((p, i) => {
                // ✅ handles both { prediction: "UP" } and { label: "Up" }
                const isUp = p.prediction === 'UP' || p.label === 'Up';
                const conf = p.confidence ?? (p.probability_up * 100);
                return (
                  <tr key={i} className={isUp ? 'row-up' : 'row-down'}>
                    <td>{i + 1}</td>
                    <td>{p.headline || p.news_text}</td>
                    <td>
                      <span className={`badge ${isUp ? 'badge-up' : 'badge-down'}`}>
                        {isUp ? '📈 UP' : '📉 DOWN'}
                      </span>
                    </td>
                    <td>{typeof conf === 'number' ? conf.toFixed(1) : '—'}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default PredictionPanel;
