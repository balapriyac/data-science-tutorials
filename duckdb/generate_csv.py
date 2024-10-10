import pandas as pd
import numpy as np

# Step 1: Generate Sales Data
data = {
    'Product_ID': np.arange(1, 101),
    'Product_Name': ['Product_' + str(i) for i in range(1, 101)],
    'Price': np.round(np.random.uniform(10, 500, 100), 2),
    'Quantity_Sold': np.random.randint(1, 100, 100),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100)
}

# Create and save sales data DataFrame
sales_df = pd.DataFrame(data)
sales_csv_file = 'sales_data.csv'
sales_df.to_csv(sales_csv_file, index=False)

# Step 2: Generate Product Details Data
product_data = {
    'Product_ID': np.arange(1, 101),  # Ensure IDs match the sales_data.csv
    'Manufacturer': ['Manufacturer_' + str(np.random.randint(1, 11)) for _ in range(100)]  # 10 different manufacturers
}

# Create and save product details DataFrame
product_details_df = pd.DataFrame(product_data)
product_csv_file = 'product_details.csv'
product_details_df.to_csv(product_csv_file, index=False)
