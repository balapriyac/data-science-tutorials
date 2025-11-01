from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class EventV1:
    """Original schema"""
    user_id: int
    event_type: str
    timestamp: datetime
    value: float
    
    def to_dict(self):
        return {
            'user_id': self.user_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'schema_version': 1
        }

@dataclass
class EventV2:
    """Schema v2: added session_id and country"""
    user_id: int
    event_type: str
    timestamp: datetime
    value: float
    session_id: Optional[str] = None
    country: Optional[str] = None
    
    def to_dict(self):
        return {
            'user_id': self.user_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'session_id': self.session_id,
            'country': self.country,
            'schema_version': 2
        }

# Now the handler that processes both versions:
class SchemaEvolutionHandler:
    """Parse and normalize multiple schema versions"""
    
    @staticmethod
    def parse_event(data: Dict[str, Any]):
        """Parse raw data into appropriate schema version"""
        version = data.get('schema_version', 1)
        
        if version == 1:
            return EventV1(
                user_id=data['user_id'],
                event_type=data['event_type'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                value=data['value']
            )
        elif version == 2:
            return EventV2(
                user_id=data['user_id'],
                event_type=data['event_type'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                value=data['value'],
                session_id=data.get('session_id'),
                country=data.get('country')
            )
        else:
            raise ValueError(f"Unknown schema version: {version}")
    
    @staticmethod
    def normalize_to_v2(event):
        """Convert any version to v2 for processing"""
        if isinstance(event, EventV1):
            # Provide defaults for new fields
            return EventV2(
                user_id=event.user_id,
                event_type=event.event_type,
                timestamp=event.timestamp,
                value=event.value,
                session_id=None,
                country='UNKNOWN'
            )
        return event

# Usage: handle mixed data
handler = SchemaEvolutionHandler()

old_data = {
    'user_id': 123,
    'event_type': 'click',
    'timestamp': '2024-01-01T00:00:00',
    'value': 50.0,
    'schema_version': 1
}

new_data = {
    'user_id': 456,
    'event_type': 'purchase',
    'timestamp': '2025-01-01T00:00:00',
    'value': 99.99,
    'session_id': 'sess_abc',
    'country': 'CA',
    'schema_version': 2
}

# Parse both formats
event_old = handler.parse_event(old_data)
event_new = handler.parse_event(new_data)

# Normalize to current version
normalized_old = handler.normalize_to_v2(event_old)
normalized_new = handler.normalize_to_v2(event_new)

print(f"Old event normalized: {normalized_old.to_dict()}")
print(f"New event: {normalized_new.to_dict()}")
