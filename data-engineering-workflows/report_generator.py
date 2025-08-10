"""
Auto Report Generator
Natural language report generation with template matching and smart caching.
"""

import json
import re
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd
import sqlite3
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReportIntent:
    """Parsed user intent for report generation"""
    metric: str
    dimension: str = None
    time_period: str = "last_week"
    filters: Dict[str, Any] = None
    aggregation: str = "sum"
    comparison: str = None

@dataclass
class ReportTemplate:
    """Report template with SQL and formatting"""
    name: str
    query_template: str
    description: str
    required_params: List[str]
    chart_type: str = "table"

class NaturalLanguageProcessor:
    """Extract intent from natural language queries"""
    
    def __init__(self):
        self.metric_patterns = {
            r'sales|revenue|income': 'sales',
            r'users?|customers?': 'users',
            r'orders?|purchases?': 'orders',
            r'engagement|activity': 'engagement'
        }
        
        self.dimension_patterns = {
            r'by region|regional': 'region',
            r'by month|monthly': 'month',
            r'by day|daily': 'day',
            r'by product|products': 'product'
        }
        
        self.time_patterns = {
            r'last week|past week': 'last_week',
            r'last month|past month': 'last_month',
            r'last quarter|past quarter': 'last_quarter',
            r'yesterday': 'yesterday',
            r'today': 'today'
        }
    
    def extract_intent(self, query: str) -> ReportIntent:
        """Parse natural language query into structured intent"""
        query_lower = query.lower()
        
        # Extract metric
        metric = 'sales'  # default
        for pattern, value in self.metric_patterns.items():
            if re.search(pattern, query_lower):
                metric = value
                break
        
        # Extract dimension
        dimension = None
        for pattern, value in self.dimension_patterns.items():
            if re.search(pattern, query_lower):
                dimension = value
                break
        
        # Extract time period
        time_period = 'last_week'  # default
        for pattern, value in self.time_patterns.items():
            if re.search(pattern, query_lower):
                time_period = value
                break
        
        # Extract comparison intent
        comparison = None
        if re.search(r'compar|vs|versus|against', query_lower):
            comparison = 'period_over_period'
        
        return ReportIntent(
            metric=metric,
            dimension=dimension,
            time_period=time_period,
            comparison=comparison
        )

class QueryOptimizer:
    """Build and optimize SQL queries from intent"""
    
    def __init__(self):
        self.date_filters = {
            'today': "DATE(created_at) = DATE('now')",
            'yesterday': "DATE(created_at) = DATE('now', '-1 day')",
            'last_week': "created_at >= DATE('now', '-7 days')",
            'last_month': "created_at >= DATE('now', '-30 days')",
            'last_quarter': "created_at >= DATE('now', '-90 days')"
        }
    
    def build_query(self, intent: ReportIntent, template: ReportTemplate) -> str:
        """Generate optimized SQL query"""
        query = template.query_template
        
        # Apply date filter
        date_filter = self.date_filters.get(intent.time_period, self.date_filters['last_week'])
        query = query.replace('{date_filter}', date_filter)
        
        # Apply grouping
        if intent.dimension:
            if intent.dimension == 'region':
                query = query.replace('{group_by}', 'GROUP BY region')
                query = query.replace('{select_dimension}', 'region,')
            elif intent.dimension == 'month':
                query = query.replace('{group_by}', "GROUP BY strftime('%Y-%m', created_at)")
                query = query.replace('{select_dimension}', "strftime('%Y-%m', created_at) as month,")
            elif intent.dimension == 'day':
                query = query.replace('{group_by}', "GROUP BY DATE(created_at)")
                query = query.replace('{select_dimension}', "DATE(created_at) as day,")
            else:
                query = query.replace('{group_by}', '')
                query = query.replace('{select_dimension}', '')
        else:
            query = query.replace('{group_by}', '')
            query = query.replace('{select_dimension}', '')
        
        return query

class ReportCache:
    """Cache report results to avoid recomputation"""
    
    def __init__(self, db_path: str = "report_cache.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize cache database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT,
                    result_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
    
    def get_cache_key(self, query: str) -> str:
        """Generate cache key from query"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def get_cached_result(self, query: str) -> Optional[pd.DataFrame]:
        """Get cached result if available and not expired"""
        cache_key = self.get_cache_key(query)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT result_data FROM cache 
                WHERE query_hash = ? AND expires_at > CURRENT_TIMESTAMP
            """, (cache_key,))
            
            row = cursor.fetchone()
            if row:
                return pd.read_json(row[0])
        
        return None
    
    def cache_result(self, query: str, result: pd.DataFrame, ttl_hours: int = 1):
        """Cache query result"""
        cache_key = self.get_cache_key(query)
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache 
                (query_hash, query, result_data, expires_at)
                VALUES (?, ?, ?, ?)
            """, (cache_key, query, result.to_json(), expires_at))

class AutoReportGenerator:
    """Main report generation system"""
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self.nl_processor = NaturalLanguageProcessor()
        self.query_optimizer = QueryOptimizer()
        self.cache = ReportCache()
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, ReportTemplate]:
        """Load report templates"""
        templates = {
            'sales': ReportTemplate(
                name="Sales Report",
                query_template="""
                    SELECT {select_dimension}
                           SUM(amount) as total_sales,
                           COUNT(*) as order_count,
                           AVG(amount) as avg_order_value
                    FROM orders 
                    WHERE {date_filter}
                    {group_by}
                    ORDER BY total_sales DESC
                """,
                description="Sales performance metrics",
                required_params=['date_filter'],
                chart_type="bar"
            ),
            'users': ReportTemplate(
                name="User Report",
                query_template="""
                    SELECT {select_dimension}
                           COUNT(*) as user_count,
                           COUNT(CASE WHEN last_login_at > DATE('now', '-30 days') THEN 1 END) as active_users
                    FROM users 
                    WHERE {date_filter}
                    {group_by}
                    ORDER BY user_count DESC
                """,
                description="User activity metrics",
                required_params=['date_filter'],
                chart_type="line"
            ),
            'engagement': ReportTemplate(
                name="Engagement Report",
                query_template="""
                    SELECT {select_dimension}
                           COUNT(*) as event_count,
                           COUNT(DISTINCT user_id) as unique_users
                    FROM events 
                    WHERE {date_filter}
                    {group_by}
                    ORDER BY event_count DESC
                """,
                description="User engagement metrics",
                required_params=['date_filter'],
                chart_type="area"
            )
        }
        return templates
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query with caching"""
        # Check cache first
        cached_result = self.cache.get_cached_result(query)
        if cached_result is not None:
            logger.info("Using cached result")
            return cached_result
        
        # Execute query
        with sqlite3.connect(self.db_path) as conn:
            result = pd.read_sql_query(query, conn)
        
        # Cache result
        self.cache.cache_result(query, result)
        return result
    
    def find_best_template(self, intent: ReportIntent) -> ReportTemplate:
        """Find the best matching template"""
        return self.templates.get(intent.metric, self.templates['sales'])
    
    def format_report(self, data: pd.DataFrame, intent: ReportIntent) -> Dict[str, Any]:
        """Format data into report structure"""
        report = {
            'title': f"{intent.metric.title()} Report",
            'generated_at': datetime.now().isoformat(),
            'period': intent.time_period,
            'data': data.to_dict('records'),
            'summary': {}
        }
        
        # Add summary statistics
        if not data.empty:
            numeric_cols = data.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                report['summary'][col] = {
                    'total': float(data[col].sum()),
                    'average': float(data[col].mean()),
                    'max': float(data[col].max()),
                    'min': float(data[col].min())
                }
        
        return report
    
    def process_report_request(self, user_request: str) -> Dict[str, Any]:
        """Main report generation pipeline"""
        try:
            # Parse intent
            intent = self.nl_processor.extract_intent(user_request)
            logger.info(f"Extracted intent: {intent}")
            
            # Find template
            template = self.find_best_template(intent)
            
            # Build query
            query = self.query_optimizer.build_query(intent, template)
            logger.info(f"Generated query: {query}")
            
            # Execute query
            data = self.execute_query(query)
            
            # Format report
            report = self.format_report(data, intent)
            
            return {
                "status": "success",
                "report": report,
                "query_used": query
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "suggestion": "Try rephrasing your request or check data availability"
            }
    
    def handle_request(self, user_input: str) -> Dict[str, Any]:
        """Process user request and return report"""
        logger.info(f"Processing request: {user_input}")
        return self.process_report_request(user_input)

def create_sample_data():
    """Create sample database for testing"""
    conn = sqlite3.connect('data.db')
    
    # Create tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            amount DECIMAL,
            region TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            email TEXT,
            region TEXT,
            last_login_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            event_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert sample data
    import random
    regions = ['North', 'South', 'East', 'West']
    
    # Sample orders
    for i in range(100):
        amount = random.randint(10, 1000)
        region = random.choice(regions)
        days_ago = random.randint(1, 30)
        conn.execute("""
            INSERT INTO orders (amount, region, created_at) 
            VALUES (?, ?, datetime('now', '-' || ? || ' days'))
        """, (amount, region, days_ago))
    
    # Sample users
    for i in range(50):
        region = random.choice(regions)
        days_ago = random.randint(1, 60)
        login_days_ago = random.randint(1, 10)
        conn.execute("""
            INSERT INTO users (email, region, created_at, last_login_at)
            VALUES (?, ?, datetime('now', '-' || ? || ' days'), datetime('now', '-' || ? || ' days'))
        """, (f'user{i}@example.com', region, days_ago, login_days_ago))
    
    # Sample events
    for i in range(200):
        user_id = random.randint(1, 50)
        event_type = random.choice(['login', 'purchase', 'view', 'click'])
        days_ago = random.randint(1, 7)
        conn.execute("""
            INSERT INTO events (user_id, event_type, created_at)
            VALUES (?, ?, datetime('now', '-' || ? || ' days'))
        """, (user_id, event_type, days_ago))
    
    conn.commit()
    conn.close()
    print("Sample database created: data.db")

if __name__ == "__main__":
    # Setup
    create_sample_data()
    
    # Create report generator
    generator = AutoReportGenerator()
    
    # Test different queries
    test_queries = [
        "Show me sales by region for last week",
        "User count by month",
        "Compare engagement metrics last month",
        "Total revenue yesterday",
        "Daily sales for last week"
    ]
    
    print("Auto Report Generator Test\n" + "=" * 40)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = generator.handle_request(query)
        
        if result['status'] == 'success':
            report = result['report']
            print(f"Title: {report['title']}")
            print(f"Period: {report['period']}")
            print(f"Records: {len(report['data'])}")
            
            # Show first few records
            if report['data']:
                print("Sample data:")
                for i, record in enumerate(report['data'][:3]):
                    print(f"  {i+1}. {record}")
            
            # Show summary
            if report['summary']:
                print("Summary:")
                for metric, stats in report['summary'].items():
                    print(f"  {metric}: Total={stats['total']:.2f}, Avg={stats['average']:.2f}")
        else:
            print(f"Error: {result['error']}")
        
        print("-" * 40)
