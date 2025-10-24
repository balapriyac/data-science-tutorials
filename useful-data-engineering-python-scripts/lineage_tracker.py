# ===================================================================
# 3. DATA LINEAGE TRACKER
# ===================================================================
"""
Traces data lineage by parsing SQL queries and building
dependency graphs for impact analysis.
"""

import re
from collections import defaultdict
import json


class DataLineageTracker:
    def __init__(self):
        """Initialize lineage tracker."""
        self.lineage_graph = defaultdict(lambda: {'upstream': set(), 'downstream': set()})
        self.transformations = {}
    
    def parse_sql_query(self, sql_query):
        """
        Parse SQL query to extract table dependencies.
        
        Args:
            sql_query: SQL query string
        
        Returns:
            dict with source and target tables
        """
        # Simple regex-based parsing (use sqlparse for production)
        sql_upper = sql_query.upper()
        
        # Extract target table (INSERT INTO, CREATE TABLE, etc.)
        target_match = re.search(r'(?:INSERT\s+INTO|CREATE\s+TABLE)\s+(\w+)', sql_upper)
        target_table = target_match.group(1) if target_match else None
        
        # Extract source tables (FROM and JOIN clauses)
        from_matches = re.findall(r'FROM\s+(\w+)', sql_upper)
        join_matches = re.findall(r'JOIN\s+(\w+)', sql_upper)
        source_tables = set(from_matches + join_matches)
        
        return {
            'target': target_table,
            'sources': list(source_tables),
            'query': sql_query
        }
    
    def add_lineage(self, source_tables, target_table, transformation_description=None):
        """
        Add lineage relationship to the graph.
        
        Args:
            source_tables: list of source table names
            target_table: target table name
            transformation_description: optional description of transformation
        """
        for source in source_tables:
            self.lineage_graph[source]['downstream'].add(target_table)
            self.lineage_graph[target_table]['upstream'].add(source)
        
        if transformation_description:
            self.transformations[target_table] = transformation_description
    
    def process_sql_file(self, sql_file_path):
        """
        Process SQL file to extract lineage.
        
        Args:
            sql_file_path: path to SQL file
        """
        with open(sql_file_path, 'r') as f:
            sql_content = f.read()
        
        # Split into individual statements (simple split by semicolon)
        statements = [s.strip() for s in sql_content.split(';') if s.strip()]
        
        for statement in statements:
            parsed = self.parse_sql_query(statement)
            if parsed['target'] and parsed['sources']:
                self.add_lineage(parsed['sources'], parsed['target'], statement)
        
        print(f"✓ Processed {sql_file_path}: {len(statements)} statements")
    
    def get_upstream_lineage(self, table_name, max_depth=10):
        """
        Get all upstream dependencies for a table.
        
        Args:
            table_name: table to trace
            max_depth: maximum depth to traverse
        
        Returns:
            set of upstream table names
        """
        upstream = set()
        to_process = [(table_name, 0)]
        visited = set()
        
        while to_process:
            current, depth = to_process.pop(0)
            
            if current in visited or depth >= max_depth:
                continue
            
            visited.add(current)
            current_upstream = self.lineage_graph[current]['upstream']
            upstream.update(current_upstream)
            
            for parent in current_upstream:
                to_process.append((parent, depth + 1))
        
        return upstream
    
    def get_downstream_lineage(self, table_name, max_depth=10):
        """
        Get all downstream dependencies for a table.
        
        Args:
            table_name: table to trace
            max_depth: maximum depth to traverse
        
        Returns:
            set of downstream table names
        """
        downstream = set()
        to_process = [(table_name, 0)]
        visited = set()
        
        while to_process:
            current, depth = to_process.pop(0)
            
            if current in visited or depth >= max_depth:
                continue
            
            visited.add(current)
            current_downstream = self.lineage_graph[current]['downstream']
            downstream.update(current_downstream)
            
            for child in current_downstream:
                to_process.append((child, depth + 1))
        
        return downstream
    
    def impact_analysis(self, table_name):
        """
        Perform impact analysis for a table.
        
        Args:
            table_name: table to analyze
        
        Returns:
            formatted impact analysis report
        """
        upstream = self.get_upstream_lineage(table_name)
        downstream = self.get_downstream_lineage(table_name)
        
        report = f"""
{'='*60}
IMPACT ANALYSIS: {table_name}
{'='*60}

UPSTREAM DEPENDENCIES ({len(upstream)}):
  Tables that {table_name} depends on:
"""
        for table in sorted(upstream):
            report += f"    ← {table}\n"
        
        report += f"\nDOWNSTREAM IMPACT ({len(downstream)}):\n"
        report += f"  Tables that will be affected if {table_name} changes:\n"
        for table in sorted(downstream):
            report += f"    → {table}\n"
        
        report += f"\n{'='*60}\n"
        
        return report
    
    def generate_mermaid_diagram(self, start_table, max_depth=3):
        """
        Generate Mermaid diagram for visualization.
        
        Args:
            start_table: table to center diagram on
            max_depth: depth of lineage to include
        
        Returns:
            Mermaid diagram string
        """
        upstream = self.get_upstream_lineage(start_table, max_depth)
        downstream = self.get_downstream_lineage(start_table, max_depth)
        all_tables = upstream | downstream | {start_table}
        
        diagram = "graph LR\n"
        
        # Add relationships
        for table in all_tables:
            for child in self.lineage_graph[table]['downstream']:
                if child in all_tables:
                    diagram += f"    {table} --> {child}\n"
        
        # Highlight the start table
        diagram += f"    style {start_table} fill:#f9f,stroke:#333,stroke-width:4px\n"
        
        return diagram
    
    def export_lineage(self, output_file='lineage.json'):
        """Export complete lineage graph to JSON."""
        # Convert sets to lists for JSON serialization
        exportable = {}
        for table, deps in self.lineage_graph.items():
            exportable[table] = {
                'upstream': list(deps['upstream']),
                'downstream': list(deps['downstream'])
            }
        
        with open(output_file, 'w') as f:
            json.dump(exportable, f, indent=2)
        
        print(f"✓ Lineage exported to {output_file}")

# Example usage
if __name__ == "__main__":
    tracker = DataLineageTracker()
    
    # Simulate adding lineage from SQL queries
    tracker.add_lineage(['raw_orders', 'raw_customers'], 'stg_orders', 
                       'Join orders with customer data')
    tracker.add_lineage(['stg_orders'], 'fact_orders', 
                       'Apply business logic and aggregations')
    tracker.add_lineage(['fact_orders', 'dim_products'], 'sales_report',
                       'Generate sales report')
    tracker.add_lineage(['raw_products'], 'dim_products',
                       'Transform product dimension')
    
    # Perform impact analysis
    print(tracker.impact_analysis('stg_orders'))
    
    # Generate diagram
    diagram = tracker.generate_mermaid_diagram('stg_orders')
    print("\nMermaid Diagram:")
    print(diagram)


