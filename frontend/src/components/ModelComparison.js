// src/components/ModelComparison.js
import React from 'react';
import './ModelComparison.css';

const ModelComparison = ({ metrics }) => {
  if (!metrics) return null;

  // Normalize: array or object { model_name: { accuracy, ... } }
  const rows = Array.isArray(metrics)
    ? metrics
    : Object.entries(metrics).map(([model, stats]) => ({
        model,
        ...(typeof stats === 'object' ? stats : { accuracy: stats }),
      }));

  if (!rows.length) return null;

  const cols = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'];

  const best = {};
  cols.forEach(c => {
    const vals = rows.map(r => r[c]).filter(v => v != null);
    best[c] = vals.length ? Math.max(...vals) : null;
  });

  const fmt = (val) => (val != null ? (val * 100).toFixed(2) + '%' : '—');

  const modelLabel = (key) => ({
    ensemble:      'Ensemble (Best)',
    random_forest: 'Random Forest',
    xgboost:       'XGBoost',
    lstm:          'LSTM',
  }[key] || key);

  return (
    <div className="mc-wrap">
      <table className="mc-table">
        <thead>
          <tr>
            <th>Model</th>
            {cols.map(c => (
              <th key={c}>{c.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              <td className="model-name">{modelLabel(row.model)}</td>
              {cols.map(c => (
                <td key={c} className={row[c] != null && row[c] === best[c] ? 'best-cell' : ''}>
                  {fmt(row[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.some(r => cols.slice(1).every(c => r[c] == null)) && (
        <p style={{ fontSize: '0.8rem', color: '#a0aec0', marginTop: '0.5rem', textAlign: 'center' }}>
          ℹ️ Only accuracy is available from the backend — add precision/recall/f1/roc_auc to <code>/models/metrics</code> endpoint for full metrics.
        </p>
      )}
    </div>
  );
};

export default ModelComparison;
