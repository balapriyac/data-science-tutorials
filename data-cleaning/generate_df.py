import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

# Create a sample dataset of customer orders
n_rows = 1000

# Generate random dates in the last year
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 3, 1)
dates = [start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days)) for _ in range(n_rows)]

# Generate customer IDs with some duplicates and inconsistent formats
customer_formats = ['CUS-{}', 'C{}', 'CUST-{}', 'Customer {}', '{}']
customer_ids = [np.random.choice(customer_formats).format(np.random.randint(1000, 9999)) for _ in range(n_rows)]

# Generate email addresses with some errors
email_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'company.com']
emails = []
for i in range(n_rows):
    username = f"user{np.random.randint(100, 999)}"
    domain = np.random.choice(email_domains)
    # Introduce some errors
    if np.random.random() < 0.05:  # Missing @ symbol
        emails.append(f"{username}{domain}")
    elif np.random.random() < 0.05:  # Extra spaces
        emails.append(f" {username}@{domain} ")
    elif np.random.random() < 0.05:  # Typos
        emails.append(f"{username}@{domain.replace('com', 'cm')}")
    else:
        emails.append(f"{username}@{domain}")

# Generate product IDs with some missing values
product_ids = [f"PROD-{np.random.randint(100, 999)}" if np.random.random() > 0.03 else np.nan for _ in range(n_rows)]

# Generate quantities with some outliers
quantities = [np.random.randint(1, 10) if np.random.random() > 0.02 else np.random.randint(100, 1000) for _ in range(n_rows)]

# Generate prices with some negative values and inconsistent formats
prices = []
for _ in range(n_rows):
    price = np.random.uniform(9.99, 199.99)
    if np.random.random() < 0.02:  # Negative price
        price = -price
    if np.random.random() < 0.1:  # String format
        prices.append(f"${price:.2f}")
    elif np.random.random() < 0.1:  # Integer format
        prices.append(int(price))
    else:
        prices.append(price)

# Generate shipping status with some inconsistent values
status_options = ['Shipped', 'shipped', 'SHIPPED', 'In Transit', 'in transit', 'In-Transit', 'Delivered', 'delivered', 'DELIVERED', 'Pending', 'pending']
shipping_status = [np.random.choice(status_options) for _ in range(n_rows)]

# Create the DataFrame
df = pd.DataFrame({
    'order_date': dates,
    'customer_id': customer_ids,
    'email': emails,
    'product_id': product_ids,
    'quantity': quantities,
    'price': prices,
    'shipping_status': shipping_status
})

# Add some completely blank rows
blank_indices = np.random.choice(range(n_rows), size=5, replace=False)
for idx in blank_indices:
    df.loc[idx, :] = np.nan

# Add some duplicate rows
dup_indices = np.random.choice(range(n_rows), size=10, replace=False)
df = pd.concat([df, df.loc[dup_indices]], ignore_index=True)

# Print the first few rows to see the data
print(df.head())
