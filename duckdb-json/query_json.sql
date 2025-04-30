-- Create a table from the JSON file
CREATE TABLE ecommerce AS 
SELECT * FROM read_json_auto('ecommerce_data.json');

-- View the data
SELECT * FROM ecommerce;

-- Count the number of orders
SELECT COUNT(*) AS order_count FROM ecommerce;

-- Get order IDs and customer names
SELECT 
    order_id,
    customer->>'name' AS customer_name
FROM ecommerce;

-- Extract customer address information
SELECT 
    order_id,
    customer->>'name' AS customer_name,
    customer->'address'->>'city' AS city,
    customer->'address'->>'state' AS state
FROM ecommerce;

-- Find orders from customers in Seattle
SELECT 
    order_id,
    customer->>'name' AS customer_name
FROM ecommerce
WHERE customer->'address'->>'city' = 'Seattle';
