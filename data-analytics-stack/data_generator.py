# ============================================================================
# data_generator.py - Generate realistic e-commerce dataset
# ============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random

def generate_ecommerce_data(n_orders=50000, n_customers=10000, n_products=500):
    """
    Generate realistic e-commerce data with proper relationships
    """
    fake = Faker()
    Faker.seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("Generating realistic e-commerce data...")
    
    # Generate customers
    customers = []
    for i in range(n_customers):
        signup_date = fake.date_between(start_date='-3y', end_date='-1d')
        customers.append({
            'customer_id': i + 1,
            'name': fake.name(),
            'email': fake.email(),
            'signup_date': signup_date,
            'country': fake.country_code(),
            'city': fake.city(),
            'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], p=[0.2, 0.5, 0.3])
        })
    
    customers_df = pd.DataFrame(customers)
    
    # Generate products
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 
                  'Sports', 'Toys', 'Beauty', 'Automotive']
    
    products = []
    for i in range(n_products):
        category = random.choice(categories)
        
        price_ranges = {
            'Electronics': (50, 2000), 'Clothing': (15, 200),
            'Home & Garden': (20, 500), 'Books': (10, 50),
            'Sports': (25, 800), 'Toys': (10, 150),
            'Beauty': (15, 200), 'Automotive': (30, 1000)
        }
        
        min_price, max_price = price_ranges[category]
        base_price = round(np.random.uniform(min_price, max_price), 2)
        
        products.append({
            'product_id': i + 1,
            'product_name': f"{category} Item {i+1}",
            'category': category,
            'base_price': base_price,
            'cost': round(base_price * 0.6, 2),
            'stock_quantity': np.random.randint(0, 1000),
            'rating': round(np.random.uniform(3.0, 5.0), 1)
        })
    
    products_df = pd.DataFrame(products)
    
    # Generate orders and order items
    orders = []
    order_items = []
    order_id = 1
    
    for _ in range(n_orders):
        customer_id = np.random.randint(1, n_customers + 1)
        customer = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
        
        order_date = fake.date_time_between(
            start_date=customer['signup_date'],
            end_date='now'
        )
        
        # Premium customers order more items
        if customer['customer_segment'] == 'Premium':
            n_items = np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.3, 0.25, 0.15, 0.1])
        else:
            n_items = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.3, 0.12, 0.05, 0.03])
        
        selected_products = np.random.choice(products_df['product_id'].values, size=n_items, replace=False)
        
        order_total = 0
        for product_id in selected_products:
            product = products_df[products_df['product_id'] == product_id].iloc[0]
            quantity = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
            
            price = product['base_price']
            if np.random.random() < 0.15:
                discount_pct = np.random.uniform(0.05, 0.25)
                price = round(price * (1 - discount_pct), 2)
            
            item_total = round(price * quantity, 2)
            order_total += item_total
            
            order_items.append({
                'order_id': order_id,
                'product_id': product_id,
                'quantity': quantity,
                'unit_price': price,
                'item_total': item_total
            })
        
        payment_status = np.random.choice(['completed', 'failed', 'pending'], p=[0.95, 0.03, 0.02])
        
        orders.append({
            'order_id': order_id,
            'customer_id': customer_id,
            'order_date': order_date,
            'order_total': round(order_total, 2),
            'shipping_cost': round(np.random.uniform(5, 20), 2),
            'payment_status': payment_status,
            'shipping_country': customer['country']
        })
        
        order_id += 1
    
    orders_df = pd.DataFrame(orders)
    order_items_df = pd.DataFrame(order_items)
    
    print(f"✓ Generated {len(customers_df):,} customers")
    print(f"✓ Generated {len(products_df):,} products")
    print(f"✓ Generated {len(orders_df):,} orders")
    print(f"✓ Generated {len(order_items_df):,} order items")
    
    return customers_df, products_df, orders_df, order_items_df


if __name__ == "__main__":
    # Generate and save data
    customers_df, products_df, orders_df, order_items_df = generate_ecommerce_data()
    
    # Save as Parquet
    # customers_df.to_parquet('customers.parquet', compression='snappy')
    # products_df.to_parquet('products.parquet', compression='snappy')
    # orders_df.to_parquet('orders.parquet', compression='snappy')
    # order_items_df.to_parquet('order_items.parquet', compression='snappy')
    
    print("\n✓ Data saved as Parquet files")
