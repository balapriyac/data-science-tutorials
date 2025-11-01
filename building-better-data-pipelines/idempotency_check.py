from hashlib import md5
from datetime import datetime

# Non-idempotent: generates different IDs each run
def process_event_bad(event):
    return {
        'id': f"{event['user_id']}_{datetime.now().timestamp()}",
        'processed_at': datetime.now(),
        'user_id': event['user_id'],
        'data': event
    }

# Idempotent: same input always produces same output
def process_event_good(event, processing_date):
    # Deterministic ID from content
    content = f"{event['user_id']}_{event['timestamp']}_{processing_date}"
    event_id = md5(content.encode()).hexdigest()
    
    return {
        'id': event_id,
        'processing_date': processing_date,  # explicit parameter
        'user_id': event['user_id'],
        'data': event
    }

# Here's how to test idempotency:
# Test that repeated runs produce identical output
def test_idempotency():
    event = {
        'user_id': 123,
        'timestamp': '2025-01-01T00:00:00',
        'value': 50
    }
    processing_date = '2025-01-01'
    
    # Run twice
    result1 = process_event_good(event, processing_date)
    result2 = process_event_good(event, processing_date)
    
    # Compare
    assert result1 == result2, "Results differ between runs"
    print("Idempotency test passed")

test_idempotency()
