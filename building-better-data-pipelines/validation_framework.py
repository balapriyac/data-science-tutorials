import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_events():
    """Generate sample event data with realistic problems"""
    base_time = datetime.now() - timedelta(days=1)
    events = []
    
    for i in range(100):
        event = {
            'user_id': i,
            'event_type': np.random.choice(['click', 'view', 'purchase']),
            'timestamp': base_time + timedelta(minutes=i*5),
            'value': np.random.randint(1, 100),
            'session_id': f'sess_{i//10}'
        }
        events.append(event)
    
    df = pd.DataFrame(events)
    
    # Inject common problems
    df.loc[5, 'user_id'] = None  # missing user
    df.loc[15, 'value'] = -50  # invalid value
    df.loc[25, 'event_type'] = 'invalid_type'  # bad category
    df.loc[35, 'timestamp'] = None  # missing timestamp
    
    return df

def validate_events(df, context=""):
    """Validate event data with detailed error reporting"""
    errors = []
    
    # Schema validation
    required_cols = ['user_id', 'event_type', 'timestamp', 'value']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Null checks
    for col in required_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_rows = df[df[col].isnull()].index.tolist()[:5]
                errors.append(
                    f"Column '{col}' has {null_count} null values "
                    f"(first null rows: {null_rows})"
                )
    
    # Range validation
    if 'value' in df.columns:
        invalid_values = df[df['value'] < 0]
        if len(invalid_values) > 0:
            bad_rows = invalid_values.index.tolist()[:5]
            errors.append(
                f"Found {len(invalid_values)} negative values "
                f"(rows: {bad_rows})"
            )
    
    # Categorical validation
    if 'event_type' in df.columns:
        valid_types = ['click', 'view', 'purchase']
        invalid_mask = ~df['event_type'].isin(valid_types)
        if invalid_mask.any():
            bad_types = df[invalid_mask]['event_type'].unique().tolist()
            count = invalid_mask.sum()
            errors.append(
                f"Found {count} invalid event_types: {bad_types}"
            )
    
    # Raise with all errors
    if errors:
        error_msg = f"Validation failed for {context}:\n"
        error_msg += "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)
    
    return df

try:
    df = create_sample_events()
    validate_events(df, context="user_events_ingestion")
    print("Validation passed")
except ValueError as e:
    print(e)



