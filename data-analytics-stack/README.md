# Data Analytics Stack

Python + Parquet + DuckDB examples from the tutorial

## Requirements
```bash
pip install duckdb pandas pyarrow numpy faker
```

## Usage
```bash
python generate_data.py
python analytics_examples.py
```

Generates sample data and runs all examples from the article.

## What It Does

- Creates realistic e-commerce dataset (50K orders)
- Saves as Parquet files (8-10x smaller than CSV)
- Queries with DuckDB (10-100x faster than pandas)
- Shows compression, joins, aggregations, performance comparisons

## Files

- `data_generator.py` - Generate e-commerce data
- `analytics_examples.py` - All tutorial examples
- `*.parquet` - Data files (auto-generated)
