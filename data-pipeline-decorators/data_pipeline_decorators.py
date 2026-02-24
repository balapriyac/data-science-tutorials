import time
import functools

def retry(max_attempts=3, delay=2, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    print(f"Attempt {attempt} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator

import sqlite3

# Setup: create an in-memory DB with some orders
conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE orders (order_id, customer_id, amount, status)")
conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", [
    (1, 101, 59.99,  "completed"),
    (2, 102, 120.00, "completed"),
    (3, 101, 34.50,  "refunded"),
    (4, 103, 89.00,  "completed"),
])
conn.commit()

# Simulate a connection that fails twice before succeeding
attempt_count = 0

@retry(max_attempts=3, delay=1, exceptions=(sqlite3.OperationalError,))
def fetch_completed_orders(conn):
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise sqlite3.OperationalError("database is locked")
    cursor = conn.execute("SELECT * FROM orders WHERE status = 'completed'")
    return cursor.fetchall()


rows = fetch_completed_orders(conn)
print(rows)

import time
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {func.__name__} completed in {elapsed:.4f}s")
        return result
    return wrapper

orders = [
    {"order_id": 1, "customer_id": 101, "amount": 59.99,  "status": "completed"},
    {"order_id": 2, "customer_id": 102, "amount": 120.00, "status": "completed"},
    {"order_id": 3, "customer_id": 101, "amount": 34.50,  "status": "refunded"},
    {"order_id": 4, "customer_id": 103, "amount": 89.00,  "status": "completed"},
    {"order_id": 5, "customer_id": 102, "amount": 45.00,  "status": "completed"},
]

@timer
def transform_orders(orders: list[dict]) -> list[dict]:
    # Compute lifetime value per customer
    ltv = {}
    for o in orders:
        if o["status"] == "completed":
            ltv[o["customer_id"]] = ltv.get(o["customer_id"], 0) + o["amount"]

    # Enrich each order with LTV and spend tier
    def spend_tier(value):
        if value >= 200: return "high"
        if value >= 100: return "mid"
        return "low"

    return [
        {**o, "customer_ltv": ltv.get(o["customer_id"], 0),
              "spend_tier": spend_tier(ltv.get(o["customer_id"], 0))}
        for o in orders
    ]


result = transform_orders(orders)

for row in result:
    print(row)

import functools

def validate_schema(required_keys):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(data, *args, **kwargs):
            if not isinstance(data, dict):
                raise TypeError(f"Expected dict, got {type(data).__name__}")
            missing = required_keys - data.keys()
            if missing:
                raise ValueError(f"Missing required keys: {missing}")
            return func(data, *args, **kwargs)
        return wrapper
    return decorator

@validate_schema(required_keys={"order_id", "customer_id", "amount", "status"})
def normalize_order(order: dict) -> dict:
    return {
        "order_id": order["order_id"],
        "customer_id": order["customer_id"],
        "amount": round(float(order["amount"]), 2),
        "status": order["status"].lower().strip(),
        "note": order.get("note", ""),   # optional — fine to be absent
    }


# Valid record — passes through
normalize_order({"order_id": 1, "customer_id": 101, "amount": "59.99", "status": "Completed"})
# {"order_id": 1, "customer_id": 101, "amount": 59.99, "status": "completed", "note": ""}

# # Missing amount — caught immediately
# normalize_order({"order_id": 2, "customer_id": 102, "status": "Completed"})
# # ValueError: Missing required keys: {'amount'}

# # Wrong type entirely — also caught
# normalize_order([1, 101, 59.99, "completed"])
# # TypeError: Expected dict, got list

import functools
import hashlib
import json

def cache_result(func):
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = hashlib.md5(
            json.dumps((args, kwargs), sort_keys=True, default=str).encode()
        ).hexdigest()

        if key not in cache:
            cache[key] = func(*args, **kwargs)
            print(f"[CACHE] Computed result for {func.__name__}")
        else:
            print(f"[CACHE] Cache hit for {func.__name__}")

        return cache[key]

    wrapper.cache_clear = lambda: cache.clear()
    return wrapper

import sqlite3

conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE discount_codes (code TEXT, discount_pct REAL, active INTEGER)")
conn.executemany("INSERT INTO discount_codes VALUES (?, ?, ?)", [
    ("SAVE10", 10.0, 1),
    ("SAVE20", 20.0, 1),
    ("OLD5",    5.0, 0),
])
conn.commit()

@cache_result
def load_discount_codes(active_only: bool = True) -> dict:
    query = "SELECT code, discount_pct FROM discount_codes"
    if active_only:
        query += " WHERE active = 1"
    rows = conn.execute(query).fetchall()
    return {code: pct for code, pct in rows}


# First call hits the database
codes = load_discount_codes(active_only=True)
# [CACHE] Computed result for load_discount_codes

# Second call is instant
codes = load_discount_codes(active_only=True)
# [CACHE] Cache hit for load_discount_codes

print(codes)
# {'SAVE10': 10.0, 'SAVE20': 20.0}

# Different argument = separate cache entry
all_codes = load_discount_codes(active_only=False)
# [CACHE] Computed result for load_discount_codes

import functools
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)

def log_step(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"START  {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"END    {func.__name__} — OK")
            return result
        except Exception as e:
            logger.error(f"FAIL   {func.__name__} — {type(e).__name__}: {e}")
            raise
    return wrapper

@log_step
def fetch_completed_orders(conn) -> list[tuple]:
    cursor = conn.execute("SELECT * FROM orders WHERE status = 'completed'")
    return cursor.fetchall()

@log_step
def transform_orders(orders: list[dict]) -> list[dict]:
    ...

@log_step
def write_results(conn, rows: list[dict]) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS orders_transformed "
                 "(order_id, customer_id, amount, status, customer_ltv, spend_tier)")
    conn.executemany(
        "INSERT INTO orders_transformed VALUES (:order_id, :customer_id, :amount, "
        ":status, :customer_ltv, :spend_tier)",
        rows,
    )
    conn.commit()
    return len(rows)


