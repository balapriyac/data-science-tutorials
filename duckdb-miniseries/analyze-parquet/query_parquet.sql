SELECT * FROM read_parquet('restaurant_orders.parquet') LIMIT 5;

DESCRIBE SELECT * FROM read_parquet('restaurant_orders.parquet') LIMIT 5;

SELECT COUNT(*) AS total_orders FROM read_parquet('restaurant_orders.parquet');

SELECT SUM(price * quantity) AS total_revenue FROM read_parquet('restaurant_orders.parquet');

SELECT menu_item, SUM(quantity) AS total_quantity
FROM read_parquet('restaurant_orders.parquet')
GROUP BY menu_item
ORDER BY total_quantity DESC
LIMIT 5;

SELECT payment_method, COUNT(*) AS order_count
FROM read_parquet('restaurant_orders.parquet')
GROUP BY payment_method
ORDER BY order_count DESC;

