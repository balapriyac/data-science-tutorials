
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
## Quick Start

### 1. ts_resampler.py

Align irregular time series data to a consistent frequency before any analysis.

```bash
# Resample to daily, fill gaps with forward-fill (default)
python ts_resampler.py --input sensor_data.csv --datetime-col "Timestamp" --freq D

# Resample to monthly, use interpolation for gaps
python ts_resampler.py --input readings.csv --datetime-col "ts" --freq MS --fill interpolate

# Per-column aggregation rules (sum sales, average temperature)
python ts_resampler.py --input data.csv --datetime-col "Date" --freq D --config agg_rules.json
```

**Frequency strings:** `T`/`min` (minutely), `H` (hourly), `D` (daily), `W` (weekly), `MS` (month start), `QS` (quarter start), `AS` (annual)

**Example `agg_rules.json`:**
```json
{
    "Sales":       "sum",
    "Temperature": "mean",
    "WindSpeed":   "max",
    "EventCount":  "count"
}
```

**Output:** `resampled.xlsx` with Resampled, Summary, and Gap Report tabs.

---

### 2. ts_anomaly_detector.py

Flag data points that deviate from expected patterns using three detection approaches.

```bash
# Z-score method (default threshold: ±3 std deviations)
python ts_anomaly_detector.py --input data.csv --datetime-col "Date" --value-cols "Sales"

# IQR method across multiple columns
python ts_anomaly_detector.py --input data.csv --datetime-col "Date" \
    --value-cols "Temperature" "Pressure" --method iqr

# Rolling window method (better for trending series)
python ts_anomaly_detector.py --input data.csv --datetime-col "ts" \
    --value-cols "Revenue" --method rolling --rolling-win 30

# Run all three methods, save anomaly charts
python ts_anomaly_detector.py --input data.csv --datetime-col "Date" \
    --value-cols "Sales" --method all --plot
```

**Methods:**
- `zscore` — flags points beyond N standard deviations from the global mean
- `iqr` — flags points outside 1.5× (configurable) the interquartile range
- `rolling` — flags points deviating from a local rolling mean; handles trends better
- `all` — runs all three and flags any point caught by any method

**Output:** Annotated Excel file with `_anomaly` flag columns; red rows = anomalies.

---

### 3. ts_decompose.py

Separate a series into its underlying components for clearer analysis.

```bash
# Monthly data with annual seasonality
python ts_decompose.py --input sales.csv --datetime-col "Date" --value-col "Revenue" --period 12

# Daily data with weekly seasonality, multiplicative model
python ts_decompose.py --input traffic.csv --datetime-col "Date" --value-col "Visits" \
    --period 7 --model multiplicative

# Resample to month-start before decomposing, save chart
python ts_decompose.py --input raw.csv --datetime-col "ts" --value-col "Value" \
    --freq MS --period 12 --plot
```

**`--model` options:**
- `additive` — use when seasonal variation is roughly constant in magnitude
- `multiplicative` — use when seasonal variation scales with the trend level

**Common `--period` values:** `7` (daily data, weekly season), `12` (monthly data, annual season), `4` (quarterly data, annual season), `24` (hourly data, daily season)

**Output:** `decomposed.xlsx` with Components and Statistics tabs; optional 4-panel chart.

---

### 4. ts_forecast.py

Generate a forward forecast using SARIMA with validation metrics.

```bash
# Basic forecast, 12 periods ahead
python ts_forecast.py --input sales.csv --datetime-col "Date" --value-col "Revenue" --periods 12

# Auto-select best model parameters
python ts_forecast.py --input sales.csv --datetime-col "Date" --value-col "Revenue" \
    --periods 6 --auto-order

# Specify SARIMA order manually, save forecast chart
python ts_forecast.py --input data.csv --datetime-col "Date" --value-col "Sales" \
    --order 1 1 1 --seasonal-order 1 1 1 12 --periods 12 --plot

# Non-seasonal ARIMA (set seasonal period to 0)
python ts_forecast.py --input data.csv --datetime-col "Date" --value-col "Value" \
    --order 2 1 2 --seasonal-order 0 0 0 0 --periods 8
```

**SARIMA parameters:**
- `--order p d q` — AR order, differencing, MA order
- `--seasonal-order P D Q s` — seasonal AR, differencing, MA, and period
- `--auto-order` — grid search over common parameter combinations, picks lowest AIC
- `--test-periods` — number of hold-out periods used for MAE/RMSE/MAPE validation

**Output:** `forecast.xlsx` with Forecast, Forecast Only, and Model Metrics tabs; optional chart. Green rows = forecast values.

---

### 5. ts_compare.py

Analyse relationships between multiple time series.

```bash
# Compare three series, resample to monthly
python ts_compare.py --input data.csv --datetime-col "Date" \
    --value-cols "Sales" "Traffic" "Conversions" --freq MS

# Cross-correlation up to 6 lag periods
python ts_compare.py --input data.csv --datetime-col "Date" \
    --value-cols "LeadMetric" "LagMetric" --max-lag 6

# Compare all numeric columns, save dual-axis charts for top pairs
python ts_compare.py --input data.csv --datetime-col "Date" --plot
```

**Output tabs:**
- **Aligned Series** — all series resampled to a shared frequency
- **Summary Stats** — mean, std, min, max, median, trend slope and direction per series
- **Pairwise Corr** — Pearson and Spearman correlations for every pair, with p-values
- **Corr Matrix** — wide-format Pearson correlation matrix
- **Lag Analysis** — peak cross-correlation and lag for each pair (identifies leading/lagging relationships)

---



