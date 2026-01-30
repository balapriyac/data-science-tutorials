# analytics_examples.py - DuckDB and Parquet analytics examples
# Code from the Modern Data Analytics Stack tutorial

import duckdb
import pandas as pd
import os
import time
from data_generator import generate_ecommerce_data


def get_file_size_mb(filename):
    """Get file size in MB"""
    return os.path.getsize(filename) / (1024 * 1024)


def main():
    print("=" * 80)
    print("MODERN DATA ANALYTICS STACK: Python + Parquet + DuckDB")
    print("=" * 80)
    
    # Generate data if not exists
    if not os.path.exists('orders.parquet'):
        print("\nGenerating sample data...")
        customers_df, products_df, orders_df, order_items_df = generate_ecommerce_data()
        
        # Save as Parquet
        customers_df.to_parquet('customers.parquet', compression='snappy')
        products_df.to_parquet('products.parquet', compression='snappy')
        orders_df.to_parquet('orders.parquet', compression='snappy')
        order_items_df.to_parquet('order_items.parquet', compression='snappy')
        
        # Save as CSV for comparison
        customers_df.to_csv('customers.csv', index=False)
        orders_df.to_csv('orders.csv', index=False)
    else:
        print("\nUsing existing Parquet files...")
        # Load for pandas comparison
        orders_df = pd.read_parquet('orders.parquet')
        order_items_df = pd.read_parquet('order_items.parquet')
        products_df = pd.read_parquet('products.parquet')
    
    # --------------------------------------------------------------------
    # Example 1: File Size Comparison
    # --------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. STORAGE EFFICIENCY: Parquet vs CSV")
    print("=" * 80)
    
    print(f"customers.csv:     {get_file_size_mb('customers.csv'):.2f} MB")
    print(f"customers.parquet: {get_file_size_mb('customers.parquet'):.2f} MB")
    print(f"Savings: {(1 - get_file_size_mb('customers.parquet')/get_file_size_mb('customers.csv'))*100:.1f}%\n")
    
    print(f"orders.csv:        {get_file_size_mb('orders.csv'):.2f} MB")
    print(f"orders.parquet:    {get_file_size_mb('orders.parquet'):.2f} MB")
    print(f"Savings: {(1 - get_file_size_mb('orders.parquet')/get_file_size_mb('orders.csv'))*100:.1f}%")
    
    # --------------------------------------------------------------------
    # Example 2: Basic DuckDB Query
    # --------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. QUERY PARQUET FILES DIRECTLY WITH DUCKDB")
    print("=" * 80)
    
    con = duckdb.connect(':memory:')
    
    query = """
    SELECT 
        customer_segment,
        COUNT(*) as num_customers,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
    FROM 'customers.parquet'
    GROUP BY customer_segment
    ORDER BY num_customers DESC
    """
    
    result = con.execute(query).fetchdf()
    print("\nCustomer Distribution:")
    print(result.to_string(index=False))
    
    # --------------------------------------------------------------------
    # Example 3: Complex Analytical Query
    # --------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. COMPLEX ANALYTICS: Monthly Revenue by Segment")
    print("=" * 80)
    
    query = """
    SELECT 
        strftime(order_date, '%Y-%m') as month,
        customer_segment,
        COUNT(DISTINCT order_id) as num_orders,
        COUNT(DISTINCT customer_id) as unique_customers,
        ROUND(SUM(order_total), 2) as total_revenue,
        ROUND(AVG(order_total), 2) as avg_order_value
    FROM 'orders.parquet'
    WHERE payment_status = 'completed'
    GROUP BY month, customer_segment
    ORDER BY month DESC, total_revenue DESC
    LIMIT 15
    """
    
    monthly_revenue = con.execute(query).fetchdf()
    print(monthly_revenue.to_string(index=False))
    
    # --------------------------------------------------------------------
    # Example 4: Multi-Table Join
    # --------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. MULTI-TABLE JOINS: Revenue by Category and Segment")
    print("=" * 80)
    
    query = """
    SELECT 
        p.category,
        o.customer_segment,
        COUNT(DISTINCT oi.order_id) as num_orders,
        SUM(oi.quantity) as units_sold,
        ROUND(SUM(oi.item_total), 2) as total_revenue,
        ROUND(AVG(oi.item_total), 2) as avg_item_value
    FROM 'order_items.parquet' oi
    JOIN 'orders.parquet' o ON oi.order_id = o.order_id
    JOIN 'products.parquet' p ON oi.product_id = p.product_id
    WHERE o.payment_status = 'completed'
    GROUP BY p.category, o.customer_segment
    ORDER BY total_revenue DESC
    LIMIT 15
    """
    
    category_analysis = con.execute(query).fetchdf()
    print(category_analysis.to_string(index=False))
    
    # --------------------------------------------------------------------
    # Example 5: Performance Comparison - DuckDB vs Pandas
    # --------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. PERFORMANCE: DuckDB vs Pandas")
    print("=" * 80)
    
    # Pandas approach
    start_time = time.time()
    merged = order_items_df.merge(orders_df, on='order_id')
    merged = merged.merge(products_df, on='product_id')
    completed = merged[merged['payment_status'] == 'completed']
    customer_patterns = completed.groupby('customer_id').agg({
        'order_id': 'nunique',
        'product_id': 'nunique',
        'item_total': ['sum', 'mean']
    })
    customer_patterns.columns = ['num_orders', 'unique_products', 'total_spent', 'avg_spent']
    customer_patterns = customer_patterns.sort_values('total_spent', ascending=False).head(100)
    pandas_time = time.time() - start_time
    
    # DuckDB approach
    start_time = time.time()
    query = """
    SELECT 
        o.customer_id,
        COUNT(DISTINCT oi.order_id) as num_orders,
        COUNT(DISTINCT oi.product_id) as unique_products,
        ROUND(SUM(oi.item_total), 2) as total_spent,
        ROUND(AVG(oi.item_total), 2) as avg_spent
    FROM 'order_items.parquet' oi
    JOIN 'orders.parquet' o ON oi.order_id = o.order_id
    WHERE o.payment_status = 'completed'
    GROUP BY o.customer_id
    ORDER BY total_spent DESC
    LIMIT 100
    """
    duckdb_result = con.execute(query).fetchdf()
    duckdb_time = time.time() - start_time
    
    print(f"\nPandas execution time:  {pandas_time:.4f} seconds")
    print(f"DuckDB execution time:  {duckdb_time:.4f} seconds")
    print(f"Speedup: {pandas_time/duckdb_time:.1f}x faster with DuckDB")
    
    print("\nTop 5 customers by total spent:")
    print(duckdb_result.head().to_string(index=False))
    
    # --------------------------------------------------------------------
    # Example 6: Reusable Analytics Function
    # --------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("6. REUSABLE ANALYTICS: Product Performance Analysis")
    print("=" * 80)
    
    def analyze_product_performance(con, category=None, min_revenue=None, top_n=10):
        """Reusable product performance analysis"""
        where_clauses = ["o.payment_status = 'completed'"]
        
        if category:
            where_clauses.append(f"p.category = '{category}'")
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        WITH product_metrics AS (
            SELECT 
                p.product_id,
                p.product_name,
                p.category,
                COUNT(DISTINCT oi.order_id) as times_ordered,
                SUM(oi.quantity) as units_sold,
                ROUND(SUM(oi.item_total), 2) as total_revenue,
                ROUND(SUM(oi.item_total) - (p.cost * SUM(oi.quantity)), 2) as profit
            FROM 'order_items.parquet' oi
            JOIN 'orders.parquet' o ON oi.order_id = o.order_id
            JOIN 'products.parquet' p ON oi.product_id = p.product_id
            WHERE {where_clause}
            GROUP BY p.product_id, p.product_name, p.category, p.cost
        )
        SELECT 
            *,
            ROUND(100.0 * profit / total_revenue, 2) as profit_margin_pct
        FROM product_metrics
        """
        
        if min_revenue:
            query += f" WHERE total_revenue >= {min_revenue}"
        
        query += f" ORDER BY total_revenue DESC LIMIT {top_n}"
        
        return con.execute(query).fetchdf()
    
    # Electronics products
    electronics = analyze_product_performance(con, category='Electronics', top_n=10)
    print("\nTop 10 Electronics Products:")
    print(electronics[['product_name', 'units_sold', 'total_revenue', 'profit_margin_pct']].to_string(index=False))
    
    # High revenue products
    print("\n\nHigh-Revenue Products (>$50k revenue):")
    high_revenue = analyze_product_performance(con, min_revenue=50000, top_n=10)
    print(high_revenue[['product_name', 'category', 'total_revenue', 'profit']].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("COMPLETE! All examples from the tutorial executed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
