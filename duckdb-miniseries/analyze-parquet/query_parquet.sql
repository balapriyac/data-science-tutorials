SELECT * FROM read_parquet('restaurant_orders.parquet') LIMIT 5;

DESCRIBE SELECT * FROM read_parquet('restaurant_orders.parquet') LIMIT 5;
