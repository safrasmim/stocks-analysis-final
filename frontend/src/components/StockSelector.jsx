import React, { useState, useEffect } from 'react';
import { API_BASE_URL, TICKERS as FALLBACK_TICKERS } from '../config/constants';

const StockSelector = ({ value, onChange }) => {
  const [tickers, setTickers] = useState(FALLBACK_TICKERS); // ✅ immediate fallback

  useEffect(() => {
    fetch(`${API_BASE_URL}/tickers`)
      .then(res => res.json())
      .then(data => {
        const list = Object.entries(data.tickers || {}).map(([code, info]) => ({
          code, name: info.name, sector: info.sector, description: info.description,
        }));
        if (list.length > 0) setTickers(list);
      })
      .catch(() => console.warn("API unreachable — using local fallback."));
  }, []);

  return (
    <select value={value} onChange={e => onChange(e.target.value)}>
      {tickers.map(t => (
        <option key={t.code} value={t.code}>
          {t.code} — {t.name}
        </option>
      ))}
    </select>
  );
};

export default StockSelector;
