"""
Automated Data Quality Monitor
Essential data quality monitoring with core features.
"""

import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityRule:
    table: str
    rule_type: str  # volume, freshness, completeness
    column: str = None
    threshold: float = None
    min_rows: int = None
    max_hours: int = None

@dataclass
class QualityResult:
    rule: QualityRule
    passed: bool
    value: Any
    message: str

class DataQualityMonitor:
    def __init__(self, db_path: str, rules: List[Dict]):
        self.db_path = db_path
        self.rules = [QualityRule(**rule) for rule in rules]
        self.failed_checks = []
    
    def run_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
    
    def check_volume(self, rule: QualityRule) -> QualityResult:
        """Check table row count"""
        query = f"SELECT COUNT(*) as count FROM {rule.table}"
        result = self.run_query(query)
        count = result.iloc[0]['count']
        
        passed = count >= rule.min_rows if rule.min_rows else True
        message = f"Row count: {count} (min: {rule.min_rows})"
        
        return QualityResult(rule, passed, count, message)
    
    def check_freshness(self, rule: QualityRule) -> QualityResult:
        """Check data recency"""
        query = f"SELECT MAX({rule.column}) as latest FROM {rule.table}"
        result = self.run_query(query)
        latest = pd.to_datetime(result.iloc[0]['latest'])
        
        hours_old = (datetime.now() - latest).total_seconds() / 3600
        passed = hours_old <= rule.max_hours
        message = f"Data age: {hours_old:.1f}h (max: {rule.max_hours}h)"
        
        return QualityResult(rule, passed, hours_old, message)
    
    def check_completeness(self, rule: QualityRule) -> QualityResult:
        """Check null percentage"""
        query = f"""
        SELECT 
            COUNT(*) as total,
            COUNT({rule.column}) as non_null
        FROM {rule.table}
        """
        result = self.run_query(query)
        total = result.iloc[0]['total']
        non_null = result.iloc[0]['non_null']
        
        completeness = non_null / total if total > 0 else 0
        passed = completeness >= rule.threshold
        message = f"Completeness: {completeness:.1%} (min: {rule.threshold:.1%})"
        
        return QualityResult(rule, passed, completeness, message)
    
    def validate_table(self, rule: QualityRule) -> QualityResult:
        """Run appropriate validation"""
        try:
            if rule.rule_type == 'volume':
                return self.check_volume(rule)
            elif rule.rule_type == 'freshness':
                return self.check_freshness(rule)
            elif rule.rule_type == 'completeness':
                return self.check_completeness(rule)
        except Exception as e:
            return QualityResult(rule, False, None, f"Check failed: {e}")
    
    def run_daily_checks(self) -> List[QualityResult]:
        """Execute all quality checks"""
        results = []
        
        for rule in self.rules:
            logger.info(f"Checking {rule.rule_type} for {rule.table}")
            result = self.validate_table(rule)
            results.append(result)
            
            if not result.passed:
                self.failed_checks.append(result)
                logger.warning(f"FAILED: {result.message}")
        
        self.generate_report(results)
        return results
    
    def generate_report(self, results: List[QualityResult]):
        """Create quality report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"quality_report_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"Data Quality Report - {datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
            
            for result in results:
                status = "PASS" if result.passed else "FAIL"
                f.write(f"{result.rule.table} - {result.rule.rule_type}: {status}\n")
                f.write(f"  {result.message}\n\n")
        
        logger.info(f"Report saved: {filename}")

# Example usage and setup
def create_sample_data():
    """Create test database"""
    conn = sqlite3.connect('test_data.db')
    
    # Create tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert test data
    conn.execute("INSERT OR REPLACE INTO users (id, email) VALUES (1, 'test@example.com')")
    conn.execute("INSERT OR REPLACE INTO users (id, email) VALUES (2, NULL)")  # Missing email
    conn.execute("INSERT OR REPLACE INTO events (user_id) VALUES (1)")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Setup
    create_sample_data()
    
    # Define quality rules
    rules = [
        {"table": "users", "rule_type": "volume", "min_rows": 1},
        {"table": "users", "rule_type": "completeness", "column": "email", "threshold": 0.8},
        {"table": "events", "rule_type": "freshness", "column": "event_time", "max_hours": 24}
    ]
    
    # Run monitoring
    monitor = DataQualityMonitor('test_data.db', rules)
    results = monitor.run_daily_checks()
    
    # Print summary
    passed = sum(1 for r in results if r.passed)
    print(f"\nResults: {passed}/{len(results)} checks passed")
    
    if monitor.failed_checks:
        print("\nFailed checks:")
        for check in monitor.failed_checks:
            print(f"  - {check.message}")
