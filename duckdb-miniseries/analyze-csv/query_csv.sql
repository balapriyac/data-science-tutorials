
SELECT * FROM read_csv('shopping_data.csv') LIMIT 5;

DESCRIBE SELECT * FROM read_csv('shopping_data.csv') LIMIT 5;

SELECT
    MIN(age) AS min_age, MAX(age) AS max_age, AVG(age) AS avg_age,
    MIN(purchase_amount) AS min_purchase, MAX(purchase_amount) AS max_purchase, AVG(purchase_amount) AS avg_purchase
FROM read_csv('shopping_data.csv');

SELECT customer_name, age, purchase_amount, category
FROM read_csv_auto('shopping_data.csv')
WHERE purchase_amount > 200
ORDER BY purchase_amount DESC;

SELECT category, COUNT(*) AS total_purchases, SUM(purchase_amount) AS total_sales, AVG(purchase_amount) AS avg_spent
FROM read_csv_auto('shopping_data.csv')
GROUP BY category
ORDER BY total_sales DESC;

