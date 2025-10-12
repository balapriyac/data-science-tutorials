
## 🎯 Overview

Data analysts spend 50-70% of their time on repetitive formatting, report preparation, and data reconciliation tasks. This toolkit reclaims that time by automating the boring but essential parts of the analyst workflow.

## 📦 Scripts Included

### 1. Automated Report Formatter
**Problem:** Manually formatting Excel reports with consistent styling, conditional formatting, and summary rows.

**Solution:** Transforms raw analysis data into polished, boardroom-ready Excel reports automatically.

**Features:**
- Professional styling with colored headers and borders
- Conditional color scales for numeric columns
- Automatic column width adjustment
- Smart number formatting (currency, percentages, numbers)
- Summary rows with totals
- Timestamped report generation

**Usage:**
```python
from report_formatter import format_report
import pandas as pd

df = pd.DataFrame({
    'Region': ['North', 'South', 'East'],
    'Revenue': [125000, 98000, 145000],
    'Growth_Rate': [0.15, 0.08, 0.22]
})

format_report(df, "q3_report.xlsx", title="Q3 Performance")
```

---

### 2. Cross-Source Data Reconciler
**Problem:** Matching records across different data sources with mismatched IDs, inconsistent naming, and various date formats.

**Solution:** Intelligently reconciles data using fuzzy matching and flexible parsing.

**Features:**
- Fuzzy string matching for names/descriptions
- Flexible date parsing from multiple formats
- Multiple ID field matching
- Confidence scoring for matches
- Flags low-confidence matches for review
- Side-by-side comparison of mismatches

**Usage:**
```python
from data_reconciler import DataReconciler

reconciler = DataReconciler(confidence_threshold=75)
result = reconciler.reconcile(
    crm_data, 
    finance_data,
    name_col='customer_name',
    id_cols=['customer_id'],
    date_cols=['contact_date']
)
```

---

### 3. Metric Dashboard Generator
**Problem:** Creating the same KPI dashboards repeatedly with updated data.

**Solution:** Generates interactive HTML dashboards with trends, comparisons, and performance indicators.

**Features:**
- Interactive Plotly visualizations
- Period-over-period change calculations
- Trend line analysis
- Multiple metrics on one dashboard
- Self-contained HTML (no dependencies)
- Automatic date formatting

**Usage:**
```python
from dashboard_generator import DashboardGenerator

dashboard = DashboardGenerator("Monthly Performance")
dashboard.generate_dashboard(
    df,
    date_col='Date',
    metric_cols=['Revenue', 'Customers', 'Orders']
)
```

---

### 4. Scheduled Data Refresher
**Problem:** Manually pulling data from databases every morning to update analysis.

**Solution:** Automatically connects to data sources, pulls fresh data, and performs transformations on schedule.

**Features:**
- Scheduled execution at specified intervals
- Database connection with retry logic
- Custom transformation pipeline
- Comprehensive logging
- Error notifications
- Timestamp tracking

**Usage:**
```python
from data_refresher import DataRefresher

refresher = DataRefresher()
engine = refresher.connect_database('postgresql://localhost/mydb')

transformations = [add_derived_metrics, filter_recent]
refresher.schedule_refresh(
    engine, 
    query="SELECT * FROM sales WHERE date >= NOW() - INTERVAL '30 days'",
    transformations=transformations,
    output_path='refreshed_sales.csv',
    interval_minutes=60
)
```

---

### 5. Bulk Chart Generator
**Problem:** Creating dozens of similar charts with consistent formatting for presentations.

**Solution:** Generates hundreds of formatted charts in seconds with consistent branding.

**Features:**
- Multiple chart types (bar, line, comparison, distribution)
- Consistent styling and branding
- Automatic trend lines
- Statistical overlays (mean, median, KDE)
- High-resolution export (300 DPI)
- Batch processing by categories

**Usage:**
```python
from chart_generator import BulkChartGenerator

generator = BulkChartGenerator(style='seaborn-v0_8-darkgrid')

config = {
    'line_charts': {
        'date': 'Date',
        'value': 'Sales',
        'group': 'Region'
    },
    'comparison_charts': {
        'categories': ['Region'],
        'values': ['Sales', 'Units', 'Profit']
    }
}

generator.generate_all_charts(df, config, output_dir='charts')
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas openpyxl fuzzywuzzy python-dateutil plotly sqlalchemy schedule matplotlib seaborn scipy
```

### Quick Start
1. Choose the script that addresses your biggest pain point
2. Copy the relevant script file to your project
3. Modify the example usage code for your specific data
4. Run and iterate until it fits your workflow

---

## 📋 Requirements

- Python 3.7+
- pandas >= 1.3.0
- openpyxl >= 3.0.0 (for Excel formatting)
- fuzzywuzzy >= 0.18.0 (for fuzzy matching)
- python-dateutil >= 2.8.0 (for flexible date parsing)
- plotly >= 5.0.0 (for interactive dashboards)
- sqlalchemy >= 1.4.0 (for database connections)
- schedule >= 1.1.0 (for scheduled tasks)
- matplotlib >= 3.4.0 (for chart generation)
- seaborn >= 0.11.0 (for styled charts)
- scipy >= 1.7.0 (for statistical functions)

---


**Made with ❤️ for Data Analysts who deserve better tools**
