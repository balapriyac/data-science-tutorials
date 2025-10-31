# Python Scripts for Data Engineers

## 🎯 Overview

Data engineers spend 40-60% of their time on operational tasks—monitoring pipelines, validating schemas, tracking lineage, and responding to alerts. This toolkit automates the operational burden so you can focus on architecting resilient, scalable data systems.

## 📦 Scripts Included

### 1. Pipeline Health Monitor
**Problem:** Manually checking dozens of ETL jobs across different systems to ensure they completed successfully.

**Solution:** Centralized monitoring of all data pipelines with automated health checks and alerting.

**Features:**
- Tracks execution status across multiple pipelines
- Calculates success rates and performance metrics
- Detects consecutive failures and performance degradation
- Identifies overdue jobs based on expected schedules
- Generates consolidated health reports
- Sends alerts on critical issues

**Usage:**
```python
from pipeline_monitor import PipelineHealthMonitor

monitor = PipelineHealthMonitor()

# Provide execution logs for your pipelines
logs = {
    'daily_sales_etl': sales_log_df,
    'hourly_inventory_sync': inventory_log_df
}

report = monitor.monitor_all_pipelines(logs)
print(report)
```

---

### 2. Schema Validator and Change Detector
**Problem:** Upstream data sources change without warning, breaking pipelines downstream.

**Solution:** Automatic schema comparison and drift detection with baseline tracking.

**Features:**
- Extracts schemas from databases or DataFrames
- Compares against stored baseline schemas
- Detects added/removed columns
- Identifies data type changes
- Tracks nullable constraint changes
- Generates detailed change reports
- Prevents breaking changes from propagating

**Usage:**
```python
from schema_validator import SchemaValidator

validator = SchemaValidator()

# Create baseline from current schema
current_schema = validator.extract_schema_from_db(engine, 'users')
validator.save_baseline(current_schema, 'users_baseline')

# Later, validate against baseline
new_schema = validator.extract_schema_from_db(engine, 'users')
report = validator.validate_and_report(new_schema, 'users_baseline')
print(report)
```

---

### 3. Data Lineage Tracker
**Problem:** No visibility into data dependencies and impact of changes across the data infrastructure.

**Solution:** Automated lineage mapping by parsing SQL and ETL code with visual dependency graphs.

**Features:**
- Parses SQL queries to extract dependencies
- Builds directed graph of data flow
- Tracks transformations at each stage
- Performs upstream/downstream impact analysis
- Generates Mermaid diagrams for visualization
- Exports lineage to JSON
- Identifies circular dependencies

**Usage:**
```python
from lineage_tracker import DataLineageTracker

tracker = DataLineageTracker()

# Add lineage relationships
tracker.add_lineage(
    ['raw_orders', 'raw_customers'], 
    'stg_orders',
    'Join orders with customer data'
)

# Perform impact analysis
print(tracker.impact_analysis('stg_orders'))

# Generate visual diagram
diagram = tracker.generate_mermaid_diagram('stg_orders')
```

---

### 4. Database Performance Analyzer
**Problem:** Database performance issues are difficult to diagnose without manual investigation.

**Solution:** Automated performance analysis with actionable optimization recommendations.

**Features:**
- Identifies slow-running queries with execution statistics
- Detects missing indexes based on sequential scan ratios
- Finds table bloat and vacuum candidates
- Locates unused indexes wasting space
- Analyzes table and index sizes
- Monitors connection pool statistics
- Generates SQL commands to fix issues
- Supports PostgreSQL and MySQL

**Usage:**
```python
from db_performance_analyzer import DatabasePerformanceAnalyzer
from sqlalchemy import create_engine

engine = create_engine('postgresql://user:password@localhost/db')
analyzer = DatabasePerformanceAnalyzer(engine)

report = analyzer.generate_performance_report()
print(report)

analyzer.export_recommendations('optimization_plan.json')
```

---

### 5. Data Quality Assertion Framework
**Problem:** Data quality checks scattered across scripts with no consistent framework or reporting.

**Solution:** Declarative assertion framework with detailed failure reporting and pipeline integration.

**Features:**
- Declarative assertion syntax
- Built-in assertions (nulls, uniqueness, ranges, foreign keys, patterns)
- Custom assertion support
- Severity levels (error/warning/info)
- Detailed failure reports with context
- Failed row extraction
- JSON export of results
- Stop-on-error mode

**Usage:**
```python
from data_quality_framework import DataQualityFramework

dq = DataQualityFramework('users_table')

# Add assertions
dq.assert_row_count_range(min_rows=1000, max_rows=1000000)
dq.assert_no_nulls(['user_id', 'email'])
dq.assert_unique('user_id')
dq.assert_value_range('age', min_value=0, max_value=120)
dq.assert_foreign_key('department_id', dept_df, 'dept_id')

# Run all checks
passed = dq.run_all_assertions(df)

# Generate report
print(dq.generate_report())

# Export results
dq.export_results('quality_report.json')
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas sqlalchemy schedule fuzzywuzzy psycopg2-binary pymysql numpy scipy
```

### Installation
```bash
git clone https://github.com/yourusername/engineer-python-scripts.git
cd engineer-python-scripts
pip install -r requirements.txt
```

### Quick Start
1. Identify your biggest operational pain point
2. Copy the relevant script to your workspace
3. Configure for your database/orchestration system
4. Integrate into your operational workflows
5. Customize thresholds and alerts for your needs

---

## 📋 Requirements

- Python 3.7+
- pandas >= 1.3.0
- sqlalchemy >= 1.4.0
- psycopg2-binary >= 2.9.0 (for PostgreSQL)
- pymysql >= 1.0.0 (for MySQL)
- fuzzywuzzy >= 0.18.0
- schedule >= 1.1.0
- numpy >= 1.21.0
- scipy >= 1.7.0

### Optional Dependencies
- apache-airflow (for DAG integration)
- slack-sdk (for Slack notifications)
- pyyaml (for YAML configuration)

---


## 🏗️ Notes

### Pipeline Health Monitor
```
Execution Logs → Health Checker → Status Report → Alert System
                       ↓
                 Metric Calculator
                       ↓
                Historical Tracking
```

### Schema Validator
```
Current Schema → Comparator → Change Detector → Report Generator
                      ↓
              Baseline Storage
```

### Lineage Tracker
```
SQL/ETL Code → Parser → Graph Builder → Impact Analyzer → Visualizer
```

### Performance Analyzer
```
Database Stats → Analyzer → Recommender → Action Generator
                    ↓
              Metric Collector
```

### Quality Framework
```
Assertions → Runner → Results Collector → Report Generator → Exporter
                ↓
         Failed Row Tracker
```

---








**Built by Data Engineers, for Data Engineers** 🛠️
