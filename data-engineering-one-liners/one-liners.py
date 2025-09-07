import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# Create streaming event data
np.random.seed(42)
events = []
for i in range(1000):
    properties = {
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet']),
        'page_path': np.random.choice(['/home', '/products', '/checkout']),
        'session_length': np.random.randint(60, 3600)
    }
    if np.random.random() > 0.7:
        properties['purchase_value'] = round(np.random.uniform(20, 300), 2)

    event = {
        'event_id': f'evt_{i}',
        'timestamp': (datetime.now() - timedelta(hours=np.random.randint(0, 72))).isoformat(),
        'user_id': f'user_{np.random.randint(100, 999)}',
        'event_type': np.random.choice(['view', 'click', 'purchase']),
        'metadata': json.dumps(properties)
    }
    events.append(event)

# Create database performance logs
db_logs = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=5000, freq='1min'),
    'operation': np.random.choice(['SELECT', 'INSERT', 'UPDATE'], 5000, p=[0.7, 0.2, 0.1]),
    'duration_ms': np.random.lognormal(mean=4, sigma=1, size=5000),
    'table_name': np.random.choice(['users', 'orders', 'products'], 5000),
    'rows_processed': np.random.poisson(lam=25, size=5000),
    'connection_id': np.random.randint(1, 20, 5000)
})

# Create API log data
api_logs = []
for i in range(800):
    log_entry = {
        'timestamp': datetime.now() - timedelta(minutes=np.random.randint(0, 1440)),
        'endpoint': np.random.choice(['/api/users', '/api/orders', '/api/metrics']),
        'status_code': np.random.choice([200, 400, 500], p=[0.8, 0.15, 0.05]),
        'response_time': np.random.exponential(150)
    }
    if log_entry['status_code'] == 200:
        log_entry['payload_size'] = np.random.randint(100, 5000)
    api_logs.append(log_entry)
