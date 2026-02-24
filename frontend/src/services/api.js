import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

api.interceptors.response.use(
  (res) => res.data,
  (err) => { console.error('API Error:', err.response?.data || err.message); throw err; }
);

const apiService = {
  health:          ()                              => api.get('/health'),
  getTickers:      ()                              => api.get('/tickers'),
  getModels:       ()                              => api.get('/models'),
  getModelMetrics: ()                              => api.get('/models/metrics'),
  predict:         (ticker, news_texts, model_type='ensemble') =>
                     api.post('/predict', { ticker, news_texts, model_type }),
};

export default apiService;
