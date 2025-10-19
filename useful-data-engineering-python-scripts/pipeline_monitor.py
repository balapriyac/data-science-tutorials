# ===================================================================
# 1. PIPELINE HEALTH MONITOR
# ===================================================================
"""
Monitors data pipeline health, tracks execution status, and alerts
on failures or performance degradation.
"""

import pandas as pd
from datetime import datetime, timedelta
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class DatabasePerformanceAnalyzer:
    def __init__(self, engine):
        """
        Initialize performance analyzer.
        
        Args:
            engine: SQLAlchemy engine connected to database
        """
        self.engine = engine
        self.db_type = engine.dialect.name
        self.recommendations = []
    
    def analyze_slow_queries(self, min_duration_seconds=1, limit=20):
        """
        Identify slow-running queries.
        
        Args:
            min_duration_seconds: minimum query duration to report
            limit: maximum number of queries to return
        
        Returns:
            DataFrame with slow queries
        """
        if self.db_type == 'postgresql':
            query = text(f"""
                SELECT 
                    query,
                    calls,
                    total_exec_time / 1000 as total_time_sec,
                    mean_exec_time / 1000 as avg_time_sec,
                    max_exec_time / 1000 as max_time_sec
                FROM pg_stat_statements
                WHERE mean_exec_time / 1000 > :min_duration
                ORDER BY total_exec_time DESC
                LIMIT :limit
            """)
            
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn, params={
                    'min_duration': min_duration_seconds,
                    'limit': limit
                })
        else:
            # Placeholder for other databases
            result = pd.DataFrame()
        
        if not result.empty:
            print(f"✓ Found {len(result)} slow queries")
            for idx, row in result.head(5).iterrows():
                self.recommendations.append({
                    'type': 'slow_query',
                    'severity': 'high',
                    'description': f"Query taking avg {row['avg_time_sec']:.2f}s",
                    'query': row['query'][:100]
                })
        
        return result
    
    def analyze_missing_indexes(self):
        """
        Identify tables that might benefit from indexes.
        
        Returns:
            DataFrame with index recommendations
        """
        if self.db_type == 'postgresql':
            query = text("""
                SELECT 
                    schemaname,
                    tablename,
                    seq_scan,
                    seq_tup_read,
                    idx_scan,
                    seq_tup_read / seq_scan as avg_seq_tup,
                    CASE 
                        WHEN seq_scan > 0 THEN (seq_tup_read::float / seq_scan)
                        ELSE 0 
                    END as rows_per_scan
                FROM pg_stat_user_tables
                WHERE seq_scan > 100
                    AND seq_tup_read / NULLIF(seq_scan, 0) > 10000
                ORDER BY seq_tup_read DESC
                LIMIT 20
            """)
            
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn)
        else:
            result = pd.DataFrame()
        
        if not result.empty:
            print(f"✓ Found {len(result)} tables with potential missing indexes")
            for idx, row in result.head(5).iterrows():
                self.recommendations.append({
                    'type': 'missing_index',
                    'severity': 'medium',
                    'table': f"{row['schemaname']}.{row['tablename']}",
                    'description': f"High sequential scans: {row['seq_scan']}, avg rows scanned: {row['rows_per_scan']:.0f}",
                    'suggestion': f"Consider adding indexes on frequently filtered columns"
                })
        
        return result
    
    def analyze_table_bloat(self):
        """
        Identify tables with significant bloat.
        
        Returns:
            DataFrame with bloated tables
        """
        if self.db_type == 'postgresql':
            query = text("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
                    n_dead_tup,
                    n_live_tup,
                    ROUND(100 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) as dead_tuple_percent,
                    last_vacuum,
                    last_autovacuum
                FROM pg_stat_user_tables
                WHERE n_dead_tup > 1000
                    AND n_dead_tup > n_live_tup * 0.1
                ORDER BY n_dead_tup DESC
                LIMIT 20
            """)
            
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn)
        else:
            result = pd.DataFrame()
        
        if not result.empty:
            print(f"✓ Found {len(result)} bloated tables")
            for idx, row in result.head(5).iterrows():
                self.recommendations.append({
                    'type': 'table_bloat',
                    'severity': 'high',
                    'table': f"{row['schemaname']}.{row['tablename']}",
                    'description': f"{row['dead_tuple_percent']:.1f}% dead tuples ({row['n_dead_tup']} rows)",
                    'suggestion': f"VACUUM ANALYZE {row['schemaname']}.{row['tablename']};"
                })
        
        return result
    
    def analyze_unused_indexes(self):
        """
        Find indexes that are never used.
        
        Returns:
            DataFrame with unused indexes
        """
        if self.db_type == 'postgresql':
            query = text("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
                    idx_scan
                FROM pg_stat_user_indexes
                WHERE idx_scan = 0
                    AND indexrelname NOT LIKE '%_pkey'
                ORDER BY pg_relation_size(indexrelid) DESC
                LIMIT 20
            """)
            
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn)
        else:
            result = pd.DataFrame()
        
        if not result.empty:
            print(f"✓ Found {len(result)} unused indexes")
            for idx, row in result.head(5).iterrows():
                self.recommendations.append({
                    'type': 'unused_index',
                    'severity': 'low',
                    'index': row['indexname'],
                    'table': f"{row['schemaname']}.{row['tablename']}",
                    'description': f"Never used, size: {row['index_size']}",
                    'suggestion': f"DROP INDEX {row['schemaname']}.{row['indexname']};"
                })
        
        return result
    
    def analyze_database_size(self):
        """
        Analyze database and table sizes.
        
        Returns:
            DataFrame with size information
        """
        if self.db_type == 'postgresql':
            query = text("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
                    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as index_size,
                    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                FROM pg_tables
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY size_bytes DESC
                LIMIT 20
            """)
            
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn)
        else:
            result = pd.DataFrame()
        
        return result
    
    def generate_performance_report(self):
        """
        Generate comprehensive performance analysis report.
        
        Returns:
            formatted report string
        """
        report = f"""
{'='*70}
DATABASE PERFORMANCE ANALYSIS REPORT
Database: {self.db_type}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

"""
        # Analyze all aspects
        print("Analyzing slow queries...")
        slow_queries = self.analyze_slow_queries()
        
        print("Analyzing missing indexes...")
        missing_indexes = self.analyze_missing_indexes()
        
        print("Analyzing table bloat...")
        bloated_tables = self.analyze_table_bloat()
        
        print("Analyzing unused indexes...")
        unused_indexes = self.analyze_unused_indexes()
        
        print("Analyzing database size...")
        size_info = self.analyze_database_size()
        
        # Generate recommendations section
        report += "RECOMMENDATIONS:\n\n"
        
        # Sort by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_recs = sorted(self.recommendations, 
                           key=lambda x: severity_order[x['severity']])
        
        for idx, rec in enumerate(sorted_recs, 1):
            severity_icon = "🔴" if rec['severity'] == 'high' else "🟡" if rec['severity'] == 'medium' else "🟢"
            report += f"{idx}. {severity_icon} [{rec['type'].upper()}] {rec['severity'].upper()}\n"
            report += f"   {rec['description']}\n"
            if 'table' in rec:
                report += f"   Table: {rec['table']}\n"
            if 'suggestion' in rec:
                report += f"   ✓ Action: {rec['suggestion']}\n"
            report += "\n"
        
        # Add summary statistics
        report += f"\nSUMMARY:\n"
        report += f"  Slow Queries: {len(slow_queries)}\n"
        report += f"  Missing Index Candidates: {len(missing_indexes)}\n"
        report += f"  Bloated Tables: {len(bloated_tables)}\n"
        report += f"  Unused Indexes: {len(unused_indexes)}\n"
        
        if not size_info.empty:
            report += f"\nTOP 5 LARGEST TABLES:\n"
            for idx, row in size_info.head(5).iterrows():
                report += f"  {row['tablename']}: {row['total_size']}\n"
        
        report += f"\n{'='*70}\n"
        
        return report


# Example usage
if __name__ == "__main__":
    # Note: Replace with your actual database connection
    # engine = create_engine('postgresql://user:password@localhost/dbname')
    # analyzer = DatabasePerformanceAnalyzer(engine)
    # report = analyzer.generate_performance_report()
    # print(report)
    
    print("Database Performance Analyzer initialized.")
    print("Connect to your database and run generate_performance_report()")


# ===================================================================
# 5. DATA QUALITY ASSERTION FRAMEWORK
# ===================================================================
"""
Provides a declarative framework for defining and running data quality
assertions with detailed failure reporting.
"""

import pandas as pd
from datetime import datetime
from typing import Callable, List, Dict, Any


class DataQualityAssertion:
    def __init__(self, name: str, description: str, check_func: Callable, 
                 severity: str = 'error'):
        """
        Define a data quality assertion.
        
        Args:
            name: unique name for the assertion
            description: human-readable description
            check_func: function that takes DataFrame and returns (passed: bool, details: dict)
            severity: 'error', 'warning', or 'info'
        """
        self.name = name
        self.description = description
        self.check_func = check_func
        self.severity = severity
    
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the assertion.
        
        Returns:
            dict with assertion results
        """
        try:
            passed, details = self.check_func(df)
            return {
                'name': self.name,
                'description': self.description,
                'severity': self.severity,
                'passed': passed,
                'details': details,
                'timestamp': datetime.now()
            }
        except Exception as e:
            return {
                'name': self.name,
                'description': self.description,
                'severity': 'error',
                'passed': False,
                'details': {'error': str(e)},
                'timestamp': datetime.now()
            }


class DataQualityFramework:
    def __init__(self, dataset_name: str):
        """
        Initialize data quality framework.
        
        Args:
            dataset_name: name of the dataset being validated
        """
        self.dataset_name = dataset_name
        self.assertions: List[DataQualityAssertion] = []
        self.results = []
    
    def add_assertion(self, assertion: DataQualityAssertion):
        """Add an assertion to the framework."""
        self.assertions.append(assertion)
    
    def assert_row_count_range(self, min_rows: int, max_rows: int = None):
        """Assert row count is within expected range."""
        def check(df):
            count = len(df)
            if max_rows:
                passed = min_rows <= count <= max_rows
                details = {
                    'actual_count': count,
                    'expected_range': f'{min_rows}-{max_rows}',
                    'status': 'pass' if passed else 'fail'
                }
            else:
                passed = count >= min_rows
                details = {
                    'actual_count': count,
                    'minimum_expected': min_rows,
                    'status': 'pass' if passed else 'fail'
                }
            return passed, details
        
        self.add_assertion(DataQualityAssertion(
            f'row_count_range_{min_rows}_{max_rows}',
            f'Row count should be between {min_rows} and {max_rows}',
            check
        ))
    
    def assert_no_nulls(self, columns: List[str]):
        """Assert specified columns have no null values."""
        def check(df):
            null_counts = {}
            all_passed = True
            
            for col in columns:
                if col in df.columns:
                    null_count = df[col].isna().sum()
                    null_counts[col] = null_count
                    if null_count > 0:
                        all_passed = False
                else:
                    null_counts[col] = 'column_not_found'
                    all_passed = False
            
            return all_passed, {'null_counts': null_counts}
        
        self.add_assertion(DataQualityAssertion(
            f'no_nulls_{"-".join(columns)}',
            f'Columns {columns} should have no null values',
            check,
            severity='error'
        ))
    
    def assert_unique(self, column: str):
        """Assert column values are unique."""
        def check(df):
            if column not in df.columns:
                return False, {'error': 'column_not_found'}
            
            total = len(df)
            unique = df[column].nunique()
            duplicates = total - unique
            passed = duplicates == 0
            
            details = {
                'total_rows': total,
                'unique_values': unique,
                'duplicate_count': duplicates
            }
            
            if not passed:
                # Find duplicate values
                dup_values = df[df.duplicated(subset=[column], keep=False)][column].unique()
                details['sample_duplicates'] = list(dup_values[:5])
            
            return passed, details
        
        self.add_assertion(DataQualityAssertion(
            f'unique_{column}',
            f'Column {column} should contain unique values',
            check,
            severity='error'
        ))
    
    def assert_value_range(self, column: str, min_value: float, max_value: float):
        """Assert numeric column values are within range."""
        def check(df):
            if column not in df.columns:
                return False, {'error': 'column_not_found'}
            
            values = df[column].dropna()
            below_min = (values < min_value).sum()
            above_max = (values > max_value).sum()
            passed = below_min == 0 and above_max == 0
            
            details = {
                'expected_range': f'{min_value}-{max_value}',
                'actual_range': f'{values.min()}-{values.max()}',
                'values_below_min': int(below_min),
                'values_above_max': int(above_max)
            }
            
            return passed, details
        
        self.add_assertion(DataQualityAssertion(
            f'range_{column}_{min_value}_{max_value}',
            f'Column {column} values should be between {min_value} and {max_value}',
            check
        ))
    
    def assert_foreign_key(self, column: str, reference_df: pd.DataFrame, 
                          reference_column: str):
        """Assert foreign key integrity."""
        def check(df):
            if column not in df.columns:
                return False, {'error': 'column_not_found'}
            
            values = set(df[column].dropna().unique())
            reference_values = set(reference_df[reference_column].dropna().unique())
            
            orphaned = values - reference_values
            passed = len(orphaned) == 0
            
            details = {
                'total_distinct_values': len(values),
                'orphaned_count': len(orphaned),
                'orphaned_sample': list(orphaned)[:10] if orphaned else []
            }
            
            return passed, details
        
        self.add_assertion(DataQualityAssertion(
            f'fk_{column}_references_{reference_column}',
            f'Foreign key {column} should reference {reference_column}',
            check,
            severity='error'
        ))
    
    def assert_custom(self, name: str, description: str, check_func: Callable, 
                     severity: str = 'error'):
        """Add a custom assertion."""
        self.add_assertion(DataQualityAssertion(name, description, check_func, severity))
    
    def run_all_assertions(self, df: pd.DataFrame) -> bool:
        """
        Run all assertions and collect results.
        
        Returns:
            True if all critical assertions passed
        """
        print(f"\nRunning {len(self.assertions)} assertions on {self.dataset_name}...")
        
        self.results = []
        all_passed = True
        
        for assertion in self.assertions:
            result = assertion.run(df)
            self.results.append(result)
            
            if not result['passed'] and result['severity'] == 'error':
                all_passed = False
        
        return all_passed
    
    def generate_report(self) -> str:
        """Generate detailed quality report."""
        if not self.results:
            return "No assertions have been run yet."
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed
        
        report = f"""
{'='*70}
DATA QUALITY REPORT
Dataset: {self.dataset_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

SUMMARY:
  Total Assertions: {total}
  Passed: {passed} ✓
  Failed: {failed} ✗
  Success Rate: {(passed/total*100):.1f}%

"""
        # Group by status
        failed_results = [r for r in self.results if not r['passed']]
        passed_results = [r for r in self.results if r['passed']]
        
        if failed_results:
            report += "FAILED ASSERTIONS:\n\n"
            for idx, result in enumerate(failed_results, 1):
                severity_icon = "🔴" if result['severity'] == 'error' else "🟡"
                report += f"{idx}. {severity_icon} {result['name']}\n"
                report += f"   Description: {result['description']}\n"
                report += f"   Severity: {result['severity'].upper()}\n"
                report += f"   Details:\n"
                for key, value in result['details'].items():
                    report += f"     - {key}: {value}\n"
                report += "\n"
        
        report += f"PASSED ASSERTIONS: {len(passed_results)}\n"
        for result in passed_results[:5]:  # Show first 5
            report += f"  ✓ {result['name']}\n"
        
        if len(passed_results) > 5:
            report += f"  ... and {len(passed_results) - 5} more\n"
        
        report += f"\n{'='*70}\n"
        
        return report
    
    def export_results(self, output_file: str = 'quality_results.json'):
        """Export results to JSON."""
        import json
        
        # Convert datetime to string for JSON serialization
        exportable = []
        for result in self.results:
            result_copy = result.copy()
            result_copy['timestamp'] = result_copy['timestamp'].isoformat()
            exportable.append(result_copy)
        
        with open(output_file, 'w') as f:
            json.dump({
                'dataset': self.dataset_name,
                'timestamp': datetime.now().isoformat(),
                'results': exportable
            }, f, indent=2)
        
        print(f"✓ Results exported to {output_file}")


# Example usage
if __name__ == "__main__":
    # Sample data
    df = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5, 6],
        'username': ['alice', 'bob', 'charlie', 'david', 'eve', 'frank'],
        'age': [25, 30, -5, 45, 150, 35],  # Some invalid ages
        'email': ['alice@example.com', None, 'charlie@example.com', 
                 'david@example.com', 'eve@example.com', 'frank@example.com'],
        'department_id': [1, 2, 1, 3, 2, 99]  # 99 doesn't exist in departments
    })
    
    departments = pd.DataFrame({
        'dept_id': [1, 2, 3],
        'dept_name': ['Engineering', 'Sales', 'Marketing']
    })
    
    # Create quality framework
    dq = DataQualityFramework('users_table')
    
    # Add assertions
    dq.assert_row_count_range(min_rows=5, max_rows=1000)
    dq.assert_no_nulls(['user_id', 'username', 'email'])
    dq.assert_unique('user_id')
    dq.assert_unique('username')
    dq.assert_value_range('age', min_value=0, max_value=120)
    dq.assert_foreign_key('department_id', departments, 'dept_id')
    
    # Run assertions
    passed = dq.run_all_assertions(df)
    
    # Generate report
    report = dq.generate_report()
    print(report)
    
    # Export results
    dq.export_results()
    
    if not passed:
        print("⚠ Data quality checks failed! Review the report above.") PipelineHealthMonitor:
    def __init__(self, config_file='pipeline_config.json'):
        """
        Initialize pipeline monitor with configuration.
        
        Args:
            config_file: JSON file with pipeline definitions and thresholds
        """
        self.config = self.load_config(config_file)
        self.health_log = []
        self.alerts = []
    
    def load_config(self, config_file):
        """Load pipeline configuration from JSON."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                'pipelines': [],
                'alert_email': 'alerts@company.com',
                'runtime_threshold_multiplier': 1.5,
                'failure_alert_threshold': 2
            }
    
    def check_pipeline_status(self, pipeline_name, execution_log_df):
        """
        Check status of a specific pipeline.
        
        Args:
            pipeline_name: name of the pipeline to check
            execution_log_df: DataFrame with columns: timestamp, status, duration_seconds
        
        Returns:
            dict with pipeline health metrics
        """
        if len(execution_log_df) == 0:
            return {
                'name': pipeline_name,
                'status': 'NO_DATA',
                'last_run': None,
                'issues': ['No execution history found']
            }
        
        # Sort by timestamp
        execution_log_df['timestamp'] = pd.to_datetime(execution_log_df['timestamp'])
        execution_log_df = execution_log_df.sort_values('timestamp', ascending=False)
        
        # Get latest execution
        latest = execution_log_df.iloc[0]
        
        # Calculate metrics
        success_rate = (execution_log_df['status'] == 'SUCCESS').sum() / len(execution_log_df)
        avg_duration = execution_log_df['duration_seconds'].mean()
        
        # Check for issues
        issues = []
        
        # Check if latest run failed
        if latest['status'] != 'SUCCESS':
            issues.append(f"Latest run FAILED at {latest['timestamp']}")
        
        # Check if overdue (no run in expected window)
        time_since_last = datetime.now() - latest['timestamp']
        expected_interval = self.config.get('expected_interval_hours', 24)
        if time_since_last > timedelta(hours=expected_interval):
            issues.append(f"No run in {time_since_last.total_seconds()/3600:.1f} hours (expected: {expected_interval}h)")
        
        # Check for performance degradation
        if latest['duration_seconds'] > avg_duration * self.config['runtime_threshold_multiplier']:
            issues.append(f"Latest run took {latest['duration_seconds']:.0f}s (avg: {avg_duration:.0f}s)")
        
        # Check consecutive failures
        consecutive_failures = 0
        for _, row in execution_log_df.iterrows():
            if row['status'] != 'SUCCESS':
                consecutive_failures += 1
            else:
                break
        
        if consecutive_failures >= self.config['failure_alert_threshold']:
            issues.append(f"{consecutive_failures} consecutive failures detected")
        
        return {
            'name': pipeline_name,
            'status': latest['status'],
            'last_run': latest['timestamp'],
            'duration': latest['duration_seconds'],
            'success_rate': success_rate,
            'avg_duration': avg_duration,
            'consecutive_failures': consecutive_failures,
            'issues': issues
        }
    
    def generate_health_report(self, pipeline_statuses):
        """Generate consolidated health report."""
        total_pipelines = len(pipeline_statuses)
        healthy = sum(1 for p in pipeline_statuses if len(p['issues']) == 0)
        failed = sum(1 for p in pipeline_statuses if p['status'] != 'SUCCESS')
        
        report = f"""
{'='*60}
PIPELINE HEALTH REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

SUMMARY:
  Total Pipelines: {total_pipelines}
  Healthy: {healthy}
  With Issues: {total_pipelines - healthy}
  Failed: {failed}

PIPELINE DETAILS:
"""
        for pipeline in pipeline_statuses:
            status_symbol = "✓" if len(pipeline['issues']) == 0 else "✗"
            report += f"\n{status_symbol} {pipeline['name']}\n"
            report += f"  Status: {pipeline['status']}\n"
            report += f"  Last Run: {pipeline['last_run']}\n"
            report += f"  Success Rate: {pipeline['success_rate']*100:.1f}%\n"
            
            if pipeline['issues']:
                report += "  ⚠ ISSUES:\n"
                for issue in pipeline['issues']:
                    report += f"    - {issue}\n"
        
        report += f"\n{'='*60}\n"
        return report
    
    def send_alert(self, report, recipients=None):
        """Send alert email with health report."""
        if recipients is None:
            recipients = [self.config['alert_email']]
        
        # Note: Configure SMTP settings for your environment
        print("📧 Alert would be sent to:", recipients)
        print(report)
        # Actual email sending code would go here
        # with proper SMTP configuration
    
    def monitor_all_pipelines(self, execution_logs):
        """
        Monitor all pipelines and generate health report.
        
        Args:
            execution_logs: dict mapping pipeline_name to execution log DataFrame
        
        Returns:
            health report string
        """
        pipeline_statuses = []
        
        for pipeline_name, log_df in execution_logs.items():
            status = self.check_pipeline_status(pipeline_name, log_df)
            pipeline_statuses.append(status)
            
            # Store in health log
            self.health_log.append({
                'timestamp': datetime.now(),
                'pipeline': pipeline_name,
                'status': status
            })
        
        report = self.generate_health_report(pipeline_statuses)
        
        # Send alerts if there are issues
        issues_found = sum(1 for p in pipeline_statuses if len(p['issues']) > 0)
        if issues_found > 0:
            self.send_alert(report)
        
        return report


# Example usage
if __name__ == "__main__":
    # Sample execution logs
    logs = {
        'daily_sales_etl': pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=10, freq='D'),
            'status': ['SUCCESS'] * 8 + ['FAILED', 'SUCCESS'],
            'duration_seconds': [300, 320, 310, 305, 315, 325, 330, 340, 350, 360]
        }),
        'hourly_inventory_sync': pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=24, freq='H'),
            'status': ['SUCCESS'] * 24,
            'duration_seconds': [60] * 24
        })
    }
    
    monitor = PipelineHealthMonitor()
    report = monitor.monitor_all_pipelines(logs)
    print(report)

