import duckdb

# View the first 5 rows of the data
duckdb.sql("SELECT * FROM 'sales_data.csv' LIMIT 5").df()

# Calculate total sales (Price * Quantity_Sold) per region
query = """
SELECT Region, SUM(Price * Quantity_Sold) as Total_Sales
FROM 'sales_data.csv'
GROUP BY Region
ORDER BY Total_Sales DESC
"""
total_sales = duckdb.sql(query).df()

print("Total sales per region:")
print(total_sales)

# Find the top 5 best-selling products by quantity
query = """
SELECT Product_Name, SUM(Quantity_Sold) as Total_Quantity
FROM 'sales_data.csv'
GROUP BY Product_Name
ORDER BY Total_Quantity DESC
LIMIT 5
"""
top_products = duckdb.sql(query).df()

print("Top 5 best-selling products:")
print(top_products)

# Calculate the average price of products by region
query = """
SELECT Region, AVG(Price) as Average_Price
FROM 'sales_data.csv'
GROUP BY Region
"""
avg_price_region = duckdb.sql(query).df()

print("Average price per region:")
print(avg_price_region)

# Calculate total quantity sold by region
query = """
SELECT Region, SUM(Quantity_Sold) as Total_Quantity
FROM 'sales_data.csv'
GROUP BY Region
ORDER BY Total_Quantity DESC
"""
total_quantity_region = duckdb.sql(query).df()

print("Total quantity sold per region:")
print(total_quantity_region)

# A simple join
query = """
SELECT s.Product_Name, s.Region, s.Price, p.Manufacturer
FROM 'sales_data.csv' s
JOIN 'product_details.csv' p
ON s.Product_ID = p.Product_ID
"""
joined_data = duckdb.sql(query).df()

print(joined_data.head())
