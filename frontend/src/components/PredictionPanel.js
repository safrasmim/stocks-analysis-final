import React, { useState } from 'react';
import apiService from '../services/api';
import './PredictionPanel.css';

const defaultHeadline = `Saudi inflation drops to 1.8% as food prices stabilize\nFederal Reserve signals potential rate cuts next quarter`;

const PredictionPanel = ({ ticker, tickerInfo }) => {
  const [newsInput, setNewsInput] = useState(defaultHeadline);
  const [modelType, setModelType] = useState('ensemble');
  const [eventDate, setEventDate] = useState(new Date().toISOString().slice(0, 10));
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    const lines = newsInput.split('\n').map((l) => l.trim()).filter(Boolean);
    if (!lines.length) return setError('Add at least one headline.');
    setLoading(true); setError(null);
    try {
      const res = await apiService.predict(ticker, lines, modelType, eventDate);
      setPredictions(res.results || []);
    } catch (e) {
      setError(e?.response?.data?.detail || 'Prediction failed. Check backend status.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="pred-panel">
      <div className="panel-head">
        <h2>Inference Console</h2>
        <p>{tickerInfo?.name || ticker} · Macro-news driven directional forecast</p>
      </div>

      <div className="controls">
        <div className="ctrl-group">
          <label>Model</label>
          <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
            <option value="ensemble">Ensemble</option>
            <option value="random_forest">Random Forest</option>
            <option value="xgboost">XGBoost</option>
            <option value="lstm">LSTM</option>
          </select>
        </div>
        <div className="ctrl-group">
          <label>Event Date (required)</label>
          <input type="date" value={eventDate} onChange={(e) => setEventDate(e.target.value)} />
        </div>
      </div>

      <label>Headlines (one per line)</label>
      <textarea rows={5} value={newsInput} onChange={(e) => setNewsInput(e.target.value)} />

      <div className="btn-row">
        <button className="btn-primary" disabled={loading} onClick={handlePredict}>{loading ? 'Running...' : 'Run Prediction'}</button>
        <button className="btn-secondary" onClick={() => setPredictions([])}>Clear Results</button>
      </div>

      {error && <div className="alert-error">⚠️ {error}</div>}

      {!!predictions.length && (
        <table>
          <thead><tr><th>Headline</th><th>Direction</th><th>Signal</th><th>Confidence</th></tr></thead>
          <tbody>
            {predictions.map((p, i) => (
              <tr key={i}>
                <td>{p.headline}</td>
                <td><span className={`badge ${p.prediction === 'UP' ? 'badge-up' : 'badge-down'}`}>{p.prediction}</span></td>
                <td>{p.signal}</td>
                <td>{p.confidence?.toFixed?.(1) ?? p.confidence}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default PredictionPanel;
