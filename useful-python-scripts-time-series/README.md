
# 5 Python Scripts for Time Series Analysis

A collection of standalone Python scripts covering the core tasks in time series analysis.

---

## Scripts

| Script | What it does |
|---|---|
| `ts_resampler.py` | Resample and aggregate irregular time series to a consistent frequency |
| `ts_anomaly_detector.py` | Detect anomalous data points using z-score, IQR, or rolling statistics |
| `ts_decompose.py` | Decompose a series into trend, seasonal, and residual components |
| `ts_forecast.py` | Fit a SARIMA model and generate a forward forecast with confidence intervals |
| `ts_compare.py` | Compare multiple series: correlation, lag analysis, summary statistics |

---

## Dependencies

| Script | Packages |
|---|---|
| `ts_resampler.py` | `pandas`, `openpyxl` |
| `ts_anomaly_detector.py` | `pandas`, `openpyxl`, `matplotlib` |
| `ts_decompose.py` | `pandas`, `openpyxl`, `statsmodels`, `matplotlib` |
| `ts_forecast.py` | `pandas`, `openpyxl`, `statsmodels`, `matplotlib` |
| `ts_compare.py` | `pandas`, `openpyxl`, `matplotlib`, `scipy` |

### Install all at once

```bash
pip install pandas openpyxl matplotlib statsmodels scipy
```

Requires **Python 3.10+**.

---


