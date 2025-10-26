# ===================================================================
# 4. DATABASE PERFORMANCE ANALYZER 
# ===================================================================
"""
Analyzes database performance and identifies optimization opportunities
including slow queries, missing indexes, and table bloat.
"""

import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime


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
        elif self.db_type == 'mysql':
            query = text("""
                SELECT 
                    DIGEST_TEXT as query,
                    COUNT_STAR as calls,
                    SUM_TIMER_WAIT/1000000000000 as total_time_sec,
                    AVG_TIMER_WAIT/1000000000000 as avg_time_sec,
                    MAX_TIMER_WAIT/1000000000000 as max_time_sec
                FROM performance_schema.events_statements_summary_by_digest
                WHERE AVG_TIMER_WAIT/1000000000000 > :min_duration
                ORDER BY SUM_TIMER_WAIT DESC
                LIMIT :limit
            """)
            
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn, params={
                    'min_duration': min_duration_seconds,
                    'limit': limit
                })
        else:
            result = pd.DataFrame()
        
        if not result.empty:
            print(f"✓ Found {len(result)} slow queries")
            for idx, row in result.head(5).iterrows():
                self.recommendations.append({
                    'type': 'slow_query',
                    'severity': 'high',
                    'description': f"Query taking avg {row['avg_time_sec']:.2f}s",
                    'query': row['query'][:100] + "..." if len(str(row['query'])) > 100 else row['query']
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
                    CASE 
                        WHEN seq_scan > 0 THEN (seq_tup_read::float / seq_scan)
                        ELSE 0 
                    END as rows_per_scan
                FROM pg_stat_user_tables
                WHERE seq_scan > 100
                    AND CASE WHEN seq_scan > 0 THEN (seq_tup_read::float / seq_scan) ELSE 0 END > 10000
                ORDER BY seq_tup_read DESC
                LIMIT 20
            """)
            
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn)
        elif self.db_type == 'mysql':
            query = text("""
                SELECT 
                    TABLE_SCHEMA as schemaname,
                    TABLE_NAME as tablename,
                    TABLE_ROWS as estimated_rows
                FROM information_schema.TABLES
                WHERE TABLE_SCHEMA NOT IN ('mysql', 'information_schema', 'performance_schema', 'sys')
                    AND TABLE_ROWS > 10000
                ORDER BY TABLE_ROWS DESC
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
                    'description': f"High sequential scans detected",
                    'suggestion': f"Analyze query patterns and consider adding indexes on frequently filtered columns"
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
                    ROUND(100 * n_dead_tup::numeric / NULLIF(n_live_tup + n_dead_tup, 0), 2) as dead_tuple_percent,
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
        elif self.db_type == 'mysql':
            query = text("""
                SELECT 
                    TABLE_SCHEMA as schemaname,
                    TABLE_NAME as tablename,
                    INDEX_NAME as indexname,
                    ROUND(STAT_VALUE * @@innodb_page_size / 1024 / 1024, 2) as size_mb
                FROM information_schema.INNODB_SYS_TABLESTATS
                WHERE TABLE_SCHEMA NOT IN ('mysql', 'information_schema', 'performance_schema', 'sys')
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
                    'description': f"Potentially unused index",
                    'suggestion': f"Consider dropping: DROP INDEX {row['schemaname']}.{row['indexname']};"
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
        elif self.db_type == 'mysql':
            query = text("""
                SELECT 
                    TABLE_SCHEMA as schemaname,
                    TABLE_NAME as tablename,
                    ROUND((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024, 2) as total_size_mb,
                    ROUND(DATA_LENGTH / 1024 / 1024, 2) as table_size_mb,
                    ROUND(INDEX_LENGTH / 1024 / 1024, 2) as index_size_mb,
                    (DATA_LENGTH + INDEX_LENGTH) as size_bytes
                FROM information_schema.TABLES
                WHERE TABLE_SCHEMA NOT IN ('mysql', 'information_schema', 'performance_schema', 'sys')
                ORDER BY size_bytes DESC
                LIMIT 20
            """)
            
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn)
        else:
            result = pd.DataFrame()
        
        return result
    
    def analyze_connection_stats(self):
        """
        Analyze database connection statistics.
        
        Returns:
            dict with connection information
        """
        stats = {}
        
        if self.db_type == 'postgresql':
            query = text("""
                SELECT 
                    COUNT(*) as total_connections,
                    COUNT(*) FILTER (WHERE state = 'active') as active,
                    COUNT(*) FILTER (WHERE state = 'idle') as idle,
                    COUNT(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction
                FROM pg_stat_activity
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchone()
                stats = {
                    'total_connections': result[0],
                    'active': result[1],
                    'idle': result[2],
                    'idle_in_transaction': result[3]
                }
        
        return stats
    
    def generate_performance_report(self):
        """
        Generate comprehensive performance analysis report.
        
        Returns:
            formatted report string
        """
        report = f"""
{'='*70}
DATABASE PERFORMANCE ANALYSIS REPORT
Database Type: {self.db_type.upper()}
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
        
        print("Analyzing connection stats...")
        connection_stats = self.analyze_connection_stats()
        
        # Generate recommendations section
        report += "RECOMMENDATIONS:\n\n"
        
        if not self.recommendations:
            report += "  ✓ No critical issues detected!\n\n"
        else:
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
                if 'index' in rec:
                    report += f"   Index: {rec['index']}\n"
                if 'suggestion' in rec:
                    report += f"   ✓ Action: {rec['suggestion']}\n"
                report += "\n"
        
        # Add summary statistics
        report += f"SUMMARY:\n"
        report += f"  Slow Queries: {len(slow_queries)}\n"
        report += f"  Missing Index Candidates: {len(missing_indexes)}\n"
        report += f"  Bloated Tables: {len(bloated_tables)}\n"
        report += f"  Unused Indexes: {len(unused_indexes)}\n"
        
        if connection_stats:
            report += f"\nCONNECTION STATISTICS:\n"
            for key, value in connection_stats.items():
                report += f"  {key.replace('_', ' ').title()}: {value}\n"
        
        if not size_info.empty:
            report += f"\nTOP 5 LARGEST TABLES:\n"
            for idx, row in size_info.head(5).iterrows():
                total_size = row.get('total_size', row.get('total_size_mb', 'N/A'))
                report += f"  {row['tablename']}: {total_size}\n"
        
        report += f"\n{'='*70}\n"
        
        return report
    
    def export_recommendations(self, output_file='performance_recommendations.json'):
        """Export recommendations to JSON file."""
        import json
        
        output = {
            'database_type': self.db_type,
            'timestamp': datetime.now().isoformat(),
            'recommendations': self.recommendations
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Recommendations exported to {output_file}")

