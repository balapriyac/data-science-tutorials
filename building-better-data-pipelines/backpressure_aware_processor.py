from collections import deque
import time

class BackpressureProcessor:
    def __init__(self, max_queue_size=1000, batch_size=100):
        self.queue = deque(maxlen=max_queue_size)
        self.batch_size = batch_size
        self.metrics = {
            'processed': 0,
            'dropped': 0,
            'queue_full_events': 0
        }
    
    def ingest(self, event):
        """Add event to queue, track drops if full"""
        if len(self.queue) >= self.queue.maxlen:
            self.metrics['dropped'] += 1
            self.metrics['queue_full_events'] += 1
            return False
        
        self.queue.append(event)
        return True
    
    def process_batch(self):
        """Process one batch from queue"""
        if not self.queue:
            return 0
        
        batch_size = min(self.batch_size, len(self.queue))
        batch = [self.queue.popleft() for _ in range(batch_size)]
        
        for event in batch:
            self._process_event(event)
        
        self.metrics['processed'] += len(batch)
        return len(batch)
    
    def _process_event(self, event):
        """Simulate actual processing"""
        time.sleep(0.01)

   def get_status(self):
        """Return operational metrics"""
        return {
            'queue_depth': len(self.queue),
            'queue_utilization': len(self.queue) / self.queue.maxlen,
            'total_processed': self.metrics['processed'],
            'total_dropped': self.metrics['dropped'],
            'queue_full_count': self.metrics['queue_full_events']
        }
    
    def check_health(self):
        """Check if system is healthy"""
        status = self.get_status()
        
        if status['queue_utilization'] > 0.8:
            return {
                'healthy': False,
                'reason': f"Queue at {status['queue_utilization']*100:.1f}% capacity"
            }
        
        if status['total_dropped'] > 0:
            return {
                'healthy': False,
                'reason': f"Dropped {status['total_dropped']} events"
            }
        
        return {'healthy': True}

processor = BackpressureProcessor(max_queue_size=500, batch_size=50)

# Simulate load
for i in range(1000):
    event = {'id': i, 'data': f'event_{i}'}
    processor.ingest(event)

# Process with monitoring
while processor.queue:
    processed = processor.process_batch()
    health = processor.check_health()
    
    if not health['healthy']:
        print(f"WARNING: {health['reason']}")

print(f"Final status: {processor.get_status()}")


