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

events_df = pd.DataFrame([{**event, **json.loads(event['metadata'])} for event in events]).drop('metadata', axis=1)
events_df

outliers = db_logs.groupby('operation').apply(lambda x: x[x['duration_ms'] > x['duration_ms'].quantile(0.95)]).reset_index(drop=True)
outliers

api_response_trends = pd.DataFrame(api_logs).set_index('timestamp').sort_index().groupby('endpoint')['response_time'].rolling('1H').mean().reset_index()
api_response_trends

schema_evolution = pd.DataFrame([{k: type(v).__name__ for k, v in json.loads(event['metadata']).items()} for event in events]).fillna('missing').nunique()
print(schema_evolution)

connection_perf = db_logs.groupby(['operation', 'connection_id']).agg({'duration_ms': ['mean', 'count'], 'rows_processed': ['sum', 'mean']}).round(2)
connection_perf

hourly_patterns = pd.DataFrame(events).assign(hour=lambda x: pd.to_datetime(x['timestamp']).dt.hour).groupby(['hour', 'event_type']).size().unstack(fill_value=0).div(pd.DataFrame(events).assign(hour=lambda x: pd.to_datetime(x['timestamp']).dt.hour).groupby('hour').size(), axis=0).round(3)
hourly_patterns

error_breakdown = pd.DataFrame(api_logs).groupby(['endpoint', 'status_code']).size().unstack(fill_value=0).div(pd.DataFrame(api_logs).groupby('endpoint').size(), axis=0).round(3)
print(error_breakdown)

anomaly_flags = db_logs.sort_values('timestamp').assign(rolling_mean=lambda x: x['duration_ms'].rolling(window=100, min_periods=10).mean()).assign(is_anomaly=lambda x: x['duration_ms'] > 2 * x['rolling_mean'])
anomaly_flags

optimized_df = db_logs.select_dtypes(include=['int', 'float']).apply(lambda x: pd.to_numeric(x, downcast='integer' if x.dtype == 'int64' else 'float')).combine_first(db_logs)
optimized_df

pipeline_metrics = pd.DataFrame(events).assign(hour=lambda x: pd.to_datetime(x['timestamp']).dt.hour).groupby('hour').agg({'event_id': 'count', 'user_id': 'nunique', 'event_type': lambda x: (x == 'purchase').mean()}).rename(columns={'event_id': 'total_events', 'user_id': 'unique_users', 'event_type': 'purchase_rate'}).round(3)
pipeline_metrics
