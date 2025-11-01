import logging
import json
import uuid
from datetime import datetime
from functools import wraps

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class StructuredLogger:
    """Emit structured, parseable log entries"""
    
    @staticmethod
    def log(level, message, **context):
        """Log with structured context"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            **context
        }
        logger.log(getattr(logging, level), json.dumps(log_entry))
    
    @classmethod
    def info(cls, message, **context):
        cls.log('INFO', message, **context)
    
    @classmethod
    def error(cls, message, **context):
        cls.log('ERROR', message, **context)
    
    @classmethod
    def warning(cls, message, **context):
        cls.log('WARNING', message, **context)

# Now add tracing to pipeline stages:
def traced_pipeline_stage(stage_name):
    """Decorator to add distributed tracing"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            correlation_id = kwargs.get('correlation_id', str(uuid.uuid4()))
            kwargs['correlation_id'] = correlation_id
            
            StructuredLogger.info(
                f"Starting {stage_name}",
                correlation_id=correlation_id,
                stage=stage_name,
                status='started'
            )
            
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                
                duration = (datetime.now() - start_time).total_seconds()
                StructuredLogger.info(
                    f"Completed {stage_name}",
                    correlation_id=correlation_id,
                    stage=stage_name,
                    status='completed',
                    duration_seconds=duration
                )
                
                return result
                
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                StructuredLogger.error(
                    f"Failed {stage_name}",
                    correlation_id=correlation_id,
                    stage=stage_name,
                    status='failed',
                    duration_seconds=duration,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
        
        return wrapper
    return decorator

# Usage: trace your pipeline stages
@traced_pipeline_stage('ingest')
def ingest_events(source, correlation_id=None):
    StructuredLogger.info(
        "Fetching from source",
        correlation_id=correlation_id,
        source=source
    )
    events = [{'id': i, 'value': i*10} for i in range(100)]
    return events

@traced_pipeline_stage('transform')
def transform_events(events, correlation_id=None):
    transformed = []
    for event in events:
        transformed.append({
            **event,
            'value_squared': event['value'] ** 2
        })
    return transformed

@traced_pipeline_stage('load')
def load_events(events, destination, correlation_id=None):
    StructuredLogger.info(
        "Loading to destination",
        correlation_id=correlation_id,
        destination=destination,
        event_count=len(events)
    )
    return len(events)

# Run pipeline with tracing
correlation_id = str(uuid.uuid4())
events = ingest_events('api.example.com', correlation_id=correlation_id)
transformed = transform_events(events, correlation_id=correlation_id)
loaded = load_events(transformed, 'warehouse', correlation_id=correlation_id)
