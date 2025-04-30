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

-- Get payment details
SELECT 
    order_id,
    payment->>'method' AS payment_method,
    CAST(payment->>'total' AS DECIMAL) AS total_amount
FROM ecommerce;

-- Unnest the items array into separate rows
SELECT 
    order_id,
    customer->>'name' AS customer_name,
    unnest(items) AS item
FROM ecommerce;

-- Get specific item details
SELECT 
    order_id,
    customer->>'name' AS customer_name,
    item->>'name' AS product_name,
    item->>'category' AS category,
    CAST(item->>'price' AS DECIMAL) AS price,
    CAST(item->>'quantity' AS INTEGER) AS quantity
FROM (
    SELECT 
        order_id,
        customer,
        unnest(items) AS item
    FROM ecommerce
) AS unnested_items;

-- Calculate total value of each order
SELECT 
    order_id,
    customer->>'name' AS customer_name,
    CAST(payment->>'total' AS DECIMAL) AS order_total,
    json_array_length(items) AS item_count
FROM ecommerce;

-- Calculate average price by product category
SELECT 
    item->>'category' AS category,
    AVG(CAST(item->>'price' AS DECIMAL)) AS avg_price
FROM (
    SELECT unnest(items) AS item
    FROM ecommerce
) AS unnested_items
GROUP BY category
ORDER BY avg_price DESC;

