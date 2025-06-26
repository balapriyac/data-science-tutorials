import pandas as pd
import numpy as np
import sqlite3
import requests
from datetime import datetime
from io import StringIO

def create_sample_csv_data():
    """Create sample data and save as CSV to simulate downloading from a source"""
    np.random.seed(42)
    data = {
        'transaction_id': [f'TXN_{i:05d}' for i in range(1000)],
        'customer_id': np.random.randint(1, 201, 1000),
        'product_name': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Headphones', 'Mouse'], 1000),
        'price': np.round(np.random.uniform(10, 1000, 1000), 2),
        'quantity': np.random.randint(1, 6, 1000),
        'transaction_date': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'customer_email': [f'user{i}@email.com' if i % 10 != 0 else None for i in range(1000)]
    }
    df = pd.DataFrame(data)
    df.to_csv('raw_transactions.csv', index=False)
    print("Sample CSV file 'raw_transactions.csv' created!")
    return 'raw_transactions.csv'

def extract_data_from_csv(csv_file_path):
    """Extract: Load data from CSV file (simulating download from external source)"""
    try:
        print(f"Extracting data from {csv_file_path}...")
        df = pd.read_csv(csv_file_path)
        print(f"Successfully extracted {len(df)} records")
        return df
    except FileNotFoundError:
        print(f"Error: {csv_file_path} not found. Creating sample data...")
        csv_file = create_sample_csv_data()
        return pd.read_csv(csv_file)

def transform_data(df):
    """Transform: Clean and enrich the data"""
    print("Transforming data...")
    
    # Start with a copy to avoid modifying original
    df_clean = df.copy()
    
    # Remove records with missing emails (data quality)
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['customer_email'])
    removed_count = initial_count - len(df_clean)
    print(f"Removed {removed_count} records with missing emails")
    
    # Calculate derived fields
    df_clean['total_amount'] = df_clean['price'] * df_clean['quantity']
    
    # Extract date components for better analysis
    df_clean['transaction_date'] = pd.to_datetime(df_clean['transaction_date'])
    df_clean['year'] = df_clean['transaction_date'].dt.year
    df_clean['month'] = df_clean['transaction_date'].dt.month
    df_clean['day_of_week'] = df_clean['transaction_date'].dt.day_name()
    
    # Create customer segments based on spending
    df_clean['customer_segment'] = pd.cut(df_clean['total_amount'], 
                                        bins=[0, 50, 200, float('inf')], 
                                        labels=['Low', 'Medium', 'High'])
    
    print(f"Transformation complete. {len(df_clean)} clean records ready")
    return df_clean

def load_data_to_sqlite(df, db_name='ecommerce_data.db', table_name='transactions'):
    """Load: Save transformed data to SQLite database"""
    print(f"Loading data to SQLite database '{db_name}'...")
    
    # Connect to SQLite database (creates if doesn't exist)
    conn = sqlite3.connect(db_name)
    
    try:
        # Load data to database
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Verify the load was successful
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        record_count = cursor.fetchone()[0]
        
        print(f"Successfully loaded {record_count} records to '{table_name}' table")
        
        # Show a sample of what was loaded
        print("\nSample of loaded data:")
        sample_df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
        print(sample_df.to_string(index=False))
        
        return f"Data successfully loaded to {db_name}"
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
        
    finally:
        conn.close()

def run_etl_pipeline():
    """Execute the complete ETL pipeline"""
    print("Starting ETL Pipeline...")
    print("=" * 50)
    
    # Extract
    raw_data = extract_data_from_csv('raw_transactions.csv')
    
    # Transform
    transformed_data = transform_data(raw_data)
    
    # Load
    load_result = load_data_to_sqlite(transformed_data)
    
    print("=" * 50)
    print("ETL Pipeline completed successfully!")
    
    return transformed_data

