# necessary imports
import pandas as pd
import seaborn as sns
from pandasql import sqldf


# Define a reusable function for running SQL queries
run_query = lambda query: sqldf(query, globals())

# Load the "tips" dataset into a Pandas DataFrame
tips_df = sns.load_dataset("tips")

# Simple select query
query_1 = """
SELECT *
FROM tips_df
LIMIT 10;
"""
result_1 = run_query(query_1)
print(result_1)

# filtering based on a condition
query_2 = """
SELECT *
FROM tips_df
WHERE total_bill > 30 AND tip > 5;
"""

result_2 = run_query(query_2)
print(result_2)

# grouping and aggregation
query_3 = """
SELECT day, AVG(total_bill) as avg_bill
FROM tips_df
GROUP BY day;
"""

result_3 = run_query(query_3)
print(result_3)


query_4 = """
SELECT day, COUNT(*) as num_transactions, AVG(total_bill) as avg_bill, MAX(tip) as max_tip
FROM tips_df
GROUP BY day;
"""

result_4 = run_query(query_4)
print(result_4)

# subqueries
query_5 = """
SELECT *
FROM tips_df
WHERE total_bill > (SELECT AVG(total_bill) FROM tips_df);
"""

result_5 = run_query(query_5)
print(result_5)

# joins

# Create another DataFrame to join with tips_df
other_data = pd.DataFrame({
    'day': ['Thur','Fri', 'Sat', 'Sun'],
    'special_event': ['Throwback Thursday', 'Feel Good Friday', 'Social Saturday','Fun Sunday', ]
})

query_6 = """
SELECT t.*, o.special_event
FROM tips_df t
LEFT JOIN other_data o ON t.day = o.day;
"""

result_6 = run_query(query_6)
print(result_6)
