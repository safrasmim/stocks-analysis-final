# Thesis Alignment Audit & Reliability Gap Analysis

## Scope audited
- Backend training/evaluation/prediction pipeline.
- Data generation and data ingestion scripts.
- Frontend capabilities versus research claims.
- Reproducibility/readiness for MSc thesis reporting.

---

## Executive verdict
Your repo shows a strong prototype, but it is **not yet aligned** with your thesis claim of a **highly reliable, real-time macroeconomic + financial-news prediction system**.

Main reason: the current pipeline is largely built on **synthetic/generated data**, uses a **random split instead of strict time-aware backtesting**, and does not yet implement production-grade reliability controls (data quality checks, drift monitoring, confidence calibration, reproducible experiment tracking).

---

## Critical alignment gaps (must fix before thesis defense)

### 1) Real-time macro/news claim vs synthetic data reality
**Observed**
- News dataset is generated from templates with injected noise (`generate_sample_data.py`).
- Macro indicators are randomly generated in both script and feature module.
- Stock data script can fall back to synthetic GBM prices.

**Impact**
- Reported accuracy is not evidence of real-world predictive power.
- External validity is weak.

**Correction**
- Replace synthetic training set with real timestamped financial news + market labels.
- Keep synthetic generation only for demos/tests and separate it clearly.
- Add `data_source` metadata and reject thesis experiments that use synthetic rows.

### 2) Methodology mismatch: temporal prediction vs random split
**Observed**
- Model training uses stratified random split (`test_size=0.20`).

**Impact**
- Leakage risk: future distribution can influence training.
- Inflated metrics compared to realistic deployment.

**Correction**
- Implement walk-forward backtesting:
  - Train window: `t0..tN`
  - Validation: `tN+1..tN+k`
  - Test: strictly later period(s)
- Report mean/std over folds and worst-fold performance.

### 3) Baseline definition does not match thesis objective
**Observed**
- `BaselineModel` is named “historical price-only” but actually trains on provided generic feature matrix (not explicit OHLCV-only pipeline).

**Impact**
- Baseline comparison in thesis can be challenged as unfair/non-compliant.

**Correction**
- Build a true historical baseline from OHLCV + technical indicators only (RSI, MACD, ATR, Bollinger, returns lags).
- Freeze exact baseline feature list in code and thesis.

### 4) Inference-time macro context is effectively dummy/static
**Observed**
- Predictor inserts hardcoded date `2026-01-01` for incoming headlines.
- If macro dataframe is absent, macro feature columns are set to `0.0`.

**Impact**
- “Real-time macroeconomic integration” is not operational at inference.

**Correction**
- Require event timestamp in API request.
- Join headline to latest available macro snapshot (`asof` merge).
- Add strict validation: reject predictions if macro freshness SLA is violated.

---

## High-priority reliability gaps

### 5) Hardcoded fallback metrics in API
**Observed**
- API can return hardcoded fallback model metrics.

**Risk**
- Dashboard may show stale/non-experimental numbers.

**Fix**
- Remove hardcoded metrics from research mode.
- Return `metrics_unavailable` explicitly if artifact missing.

### 6) No confidence calibration / abstention policy
**Observed**
- Predictions always returned; no reject option for low confidence.

**Risk**
- Unsafe overconfident decisions.

**Fix**
- Add calibration (Platt/Isotonic) on validation fold.
- Add abstain zone (e.g., 0.45–0.55) and report coverage vs accuracy.

### 7) No drift monitoring
**Observed**
- No PSI/feature drift or label drift checks.

**Fix**
- Add scheduled drift report over incoming data.
- Trigger retraining or alert when thresholds exceeded.

### 8) Missing experiment tracking/governance
**Observed**
- No explicit run registry (config hash, data hash, commit SHA, seed, metrics).

**Fix**
- Add lightweight experiment log (MLflow/W&B or structured JSONL).
- Include reproducibility manifest in thesis appendix.

---

## Medium-priority scientific/reporting gaps

1. **No uncertainty-aware evaluation** (only point metrics). Add bootstrap confidence intervals.
2. **No cost-sensitive metrics**. Add directional PnL proxy, hit-rate by volatility regime, drawdown-aware analysis.
3. **Limited regional/multi-source coverage**. Add explicit source diversity and source-quality scoring.
4. **No ablation matrix**. Quantify contribution of each block (sentiment/topics/entities/macro/events).
5. **Potential class/sector imbalance**. Add per-sector confusion matrix and macro-regime breakdown.

---

## Concrete repo corrections to implement next (recommended order)

### Sprint A — Evidence integrity (1–2 weeks)
1. Introduce `data/contracts.py` with schema checks (nulls, duplicate IDs, timestamp monotonicity, timezone).
2. Separate datasets:
   - `data/raw/real_news/*.parquet`
   - `data/raw/synthetic/*.csv`
3. Add `--allow-synthetic` flag defaulting to false for research training scripts.
4. Remove hardcoded fallback metrics in API research mode.

### Sprint B — Methodological validity (1–2 weeks)
1. Replace random split with walk-forward evaluator.
2. Implement true OHLCV baseline pipeline and keep separate artifacts.
3. Add calibrated probabilities and abstain option.
4. Extend metrics report with CIs and per-regime slices.

### Sprint C — Reliability/operations (1–2 weeks)
1. Add data freshness checks and prediction refusal when stale.
2. Add drift monitor job + alert thresholds.
3. Add experiment manifest with commit SHA, data hash, seeds.
4. Add model card template and known limitations section.

---

## Thesis-ready experiment package checklist
- [ ] Real-only training/evaluation datasets with provenance table.
- [ ] Time-aware backtesting protocol and fold definitions.
- [ ] True historical baseline vs macro-news model vs hybrid model.
- [ ] Ablation study (remove each feature family one at a time).
- [ ] Probability calibration + decision-threshold analysis.
- [ ] Robustness tests (high-volatility periods, geopolitical shock windows).
- [ ] Reproducibility bundle (commands, seeds, commit IDs, artifacts).

---

## Suggested chapter-level updates in your report
1. **Methodology chapter**: explicitly define data leakage controls and temporal validation.
2. **Experiments chapter**: add ablation table + calibration plots + drift findings.
3. **Discussion chapter**: clearly separate prototype assumptions from deployable guarantees.
4. **Limitations chapter**: mention source bias, delayed macro publications, and causality limits.

---

## Bottom line
You already have a meaningful scaffold (API, feature engineering modules, multi-model training, dashboard). To make this MSc-thesis-grade and defensible, prioritize:
1) **real data provenance**,
2) **time-correct evaluation**,
3) **true baseline fairness**, and
4) **operational reliability controls**.

If these four are implemented rigorously, your project can become both publishable and credible for high-stakes evaluation.
