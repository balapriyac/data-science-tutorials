import sqlite3
import time
import functools
import hashlib
import json
import logging

# --- decorators (retry, timer, validate_schema, cache_result, log_step) ---
# ... defined above ...

# --- setup ---
conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE orders (order_id, customer_id, amount, status)")
conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", [
    (1, 101, 59.99,  "completed"),
    (2, 102, 120.00, "completed"),
    (3, 101, 34.50,  "refunded"),
    (4, 103, 89.00,  "completed"),
    (5, 102, 45.00,  "completed"),
])
conn.execute("CREATE TABLE discount_codes (code TEXT, discount_pct REAL, active INTEGER)")
conn.executemany("INSERT INTO discount_codes VALUES (?, ?, ?)", [
    ("SAVE10", 10.0, 1),
    ("SAVE20", 20.0, 1),
])
conn.commit()

# --- pipeline ---
@log_step
@retry(max_attempts=3, delay=1, exceptions=(sqlite3.OperationalError,))
def fetch_completed_orders(conn) -> list[tuple]:
    cursor = conn.execute("SELECT * FROM orders WHERE status = 'completed'")
    return cursor.fetchall()

@cache_result
def load_discount_codes(active_only: bool = True) -> dict:
    query = "SELECT code, discount_pct FROM discount_codes"
    if active_only:
        query += " WHERE active = 1"
    rows = conn.execute(query).fetchall()
    return {code: pct for code, pct in rows}

@validate_schema(required_keys={"order_id", "customer_id", "amount", "status"})
def normalize_order(order: dict) -> dict:
    return {
        "order_id": order["order_id"],
        "customer_id": order["customer_id"],
        "amount": round(float(order["amount"]), 2),
        "status": order["status"].lower().strip(),
        "note": order.get("note", ""),
    }

@log_step
@timer
def transform_orders(orders: list[dict]) -> list[dict]:
    ltv = {}
    for o in orders:
        ltv[o["customer_id"]] = ltv.get(o["customer_id"], 0) + o["amount"]

    def spend_tier(v):
        if v >= 200: return "high"
        if v >= 100: return "mid"
        return "low"

    return [
        {**o, "customer_ltv": ltv[o["customer_id"]],
              "spend_tier": spend_tier(ltv[o["customer_id"]])}
        for o in orders
    ]

@log_step
def write_results(conn, rows: list[dict]) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS orders_transformed "
                 "(order_id, customer_id, amount, status, note, customer_ltv, spend_tier)")
    conn.executemany(
        "INSERT INTO orders_transformed VALUES "
        "(:order_id, :customer_id, :amount, :status, :note, :customer_ltv, :spend_tier)",
        rows,
    )
    conn.commit()
    return len(rows)


if __name__ == "__main__":
    raw = fetch_completed_orders(conn)
    records = [normalize_order({"order_id": r[0], "customer_id": r[1],
                                 "amount": r[2], "status": r[3]}) for r in raw]
    transformed = transform_orders(records)
    n = write_results(conn, transformed)
    print(f"Wrote {n} rows")

