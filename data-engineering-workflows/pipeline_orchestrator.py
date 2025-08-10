"""
Smart Pipeline Orchestrator
Intelligent pipeline scheduling with dependency management and smart retry logic.
"""

import json
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Callable, Any
from enum import Enum
import threading
from queue import Queue
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Status(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class Pipeline:
    name: str
    function: Callable
    dependencies: List[str]
    max_retries: int = 3
    retry_delay: int = 60  # seconds

@dataclass
class Execution:
    pipeline_name: str
    status: Status
    start_time: datetime
    end_time: datetime = None
    error: str = None
    retry_count: int = 0

class SmartOrchestrator:
    def __init__(self, db_path: str = "orchestrator.db"):
        self.pipelines = {}
        self.executions = {}
        self.db_path = db_path
        self.task_queue = Queue()
        self.running = False
        self._init_db()
    
    def _init_db(self):
        """Initialize execution history database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    id TEXT PRIMARY KEY,
                    pipeline_name TEXT,
                    status TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    error TEXT,
                    retry_count INTEGER
                )
            """)
    
    def register_pipeline(self, name: str, function: Callable, dependencies: List[str] = None, max_retries: int = 3):
        """Register a new pipeline"""
        self.pipelines[name] = Pipeline(
            name=name,
            function=function,
            dependencies=dependencies or [],
            max_retries=max_retries
        )
        logger.info(f"Registered pipeline: {name}")
    
    def prerequisites_satisfied(self, pipeline_name: str) -> bool:
        """Check if all dependencies completed successfully"""
        pipeline = self.pipelines[pipeline_name]
        
        for dep in pipeline.dependencies:
            if dep not in self.executions:
                return False
            if self.executions[dep].status != Status.SUCCESS:
                return False
        
        return True
    
    def classify_failure(self, error: str) -> str:
        """Classify failure type for smart retry logic"""
        error_lower = error.lower()
        
        if any(term in error_lower for term in ['timeout', 'connection', 'network']):
            return 'transient'
        elif any(term in error_lower for term in ['data', 'validation', 'quality']):
            return 'data_quality'
        else:
            return 'unknown'
    
    def should_retry(self, execution: Execution) -> bool:
        """Determine if pipeline should be retried"""
        pipeline = self.pipelines[execution.pipeline_name]
        
        if execution.retry_count >= pipeline.max_retries:
            return False
        
        failure_type = self.classify_failure(execution.error or "")
        
        # Always retry transient failures, be cautious with data quality issues
        if failure_type == 'transient':
            return True
        elif failure_type == 'data_quality' and execution.retry_count < 1:
            return True
        
        return False
    
    def execute_pipeline(self, pipeline_name: str) -> Execution:
        """Execute a single pipeline"""
        pipeline = self.pipelines[pipeline_name]
        execution_id = f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        execution = Execution(
            pipeline_name=pipeline_name,
            status=Status.RUNNING,
            start_time=datetime.now()
        )
        
        self.executions[pipeline_name] = execution
        logger.info(f"Starting pipeline: {pipeline_name}")
        
        try:
            # Execute the pipeline function
            result = pipeline.function()
            
            execution.status = Status.SUCCESS
            execution.end_time = datetime.now()
            
            logger.info(f"Pipeline completed: {pipeline_name}")
            
        except Exception as e:
            execution.status = Status.FAILED
            execution.end_time = datetime.now()
            execution.error = str(e)
            
            logger.error(f"Pipeline failed: {pipeline_name} - {e}")
        
        # Save to database
        self._save_execution(execution_id, execution)
        return execution
    
    def _save_execution(self, execution_id: str, execution: Execution):
        """Save execution result to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO executions 
                (id, pipeline_name, status, start_time, end_time, error, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                execution_id,
                execution.pipeline_name,
                execution.status.value,
                execution.start_time,
                execution.end_time,
                execution.error,
                execution.retry_count
            ))
    
    def schedule_pipeline(self, pipeline_name: str):
        """Schedule a pipeline for execution"""
        if pipeline_name not in self.pipelines:
            logger.error(f"Pipeline not found: {pipeline_name}")
            return
        
        if not self.prerequisites_satisfied(pipeline_name):
            logger.info(f"Prerequisites not met for {pipeline_name}, scheduling retry")
            # Schedule for retry later
            threading.Timer(60, lambda: self.schedule_pipeline(pipeline_name)).start()
            return
        
        self.task_queue.put(pipeline_name)
        logger.info(f"Scheduled pipeline: {pipeline_name}")
    
    def handle_failure(self, execution: Execution):
        """Handle pipeline failure with smart retry logic"""
        if self.should_retry(execution):
            execution.retry_count += 1
            execution.status = Status.RETRYING
            
            delay = execution.retry_count * 60  # Exponential backoff
            logger.info(f"Retrying {execution.pipeline_name} in {delay}s (attempt {execution.retry_count})")
            
            threading.Timer(delay, lambda: self.schedule_pipeline(execution.pipeline_name)).start()
        else:
            logger.error(f"Pipeline {execution.pipeline_name} failed permanently after {execution.retry_count} retries")
    
    def worker(self):
        """Worker thread to process pipeline queue"""
        while self.running:
            try:
                pipeline_name = self.task_queue.get(timeout=1)
                execution = self.execute_pipeline(pipeline_name)
                
                if execution.status == Status.FAILED:
                    self.handle_failure(execution)
                
                self.task_queue.task_done()
                
            except:
                continue
    
    def start(self):
        """Start the orchestrator"""
        self.running = True
        worker_thread = threading.Thread(target=self.worker)
        worker_thread.daemon = True
        worker_thread.start()
        logger.info("Orchestrator started")
    
    def stop(self):
        """Stop the orchestrator"""
        self.running = False
        logger.info("Orchestrator stopped")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of recent executions"""
        summary = {
            'total_pipelines': len(self.pipelines),
            'recent_executions': []
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT pipeline_name, status, start_time, error
                FROM executions
                ORDER BY start_time DESC
                LIMIT 10
            """)
            
            for row in cursor:
                summary['recent_executions'].append({
                    'pipeline': row[0],
                    'status': row[1],
                    'time': row[2],
                    'error': row[3]
                })
        
        return summary

# Example pipeline functions
def extract_data():
    """Sample data extraction pipeline"""
    logger.info("Extracting data...")
    time.sleep(2)  # Simulate work
    return {"status": "extracted", "records": 1000}

def transform_data():
    """Sample data transformation pipeline"""
    logger.info("Transforming data...")
    time.sleep(3)  # Simulate work
    return {"status": "transformed", "records": 950}

def load_data():
    """Sample data loading pipeline"""
    logger.info("Loading data...")
    time.sleep(1)  # Simulate work
    return {"status": "loaded", "records": 950}

def failing_pipeline():
    """Sample pipeline that fails sometimes"""
    import random
    if random.random() < 0.3:  # 30% chance of failure
        raise Exception("Random transient failure")
    return {"status": "success"}

if __name__ == "__main__":
    # Create orchestrator
    orchestrator = SmartOrchestrator()
    
    # Register pipelines with dependencies
    orchestrator.register_pipeline("extract", extract_data)
    orchestrator.register_pipeline("transform", transform_data, dependencies=["extract"])
    orchestrator.register_pipeline("load", load_data, dependencies=["transform"])
    orchestrator.register_pipeline("cleanup", failing_pipeline, dependencies=["load"])
    
    # Start orchestrator
    orchestrator.start()
    
    # Schedule the pipeline chain
    orchestrator.schedule_pipeline("extract")
    
    # Let it run for a bit
    time.sleep(15)
    
    # Print summary
    summary = orchestrator.get_execution_summary()
    print(f"\nExecution Summary:")
    print(f"Total pipelines: {summary['total_pipelines']}")
    print("\nRecent executions:")
    for exec in summary['recent_executions']:
        print(f"  {exec['pipeline']}: {exec['status']} at {exec['time']}")
        if exec['error']:
            print(f"    Error: {exec['error']}")
    
    orchestrator.stop()
