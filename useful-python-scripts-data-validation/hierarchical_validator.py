"""
Hierarchical Relationship Validator
Validates graph and tree structures in relational data
"""

import pandas as pd
import numpy as np
import json
import argparse
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')


class HierarchyValidator:
    """Validates hierarchical and graph relationships in data"""
    
    def __init__(self, filepath: str, node_id_col: str, parent_id_col: str,
                 max_depth: Optional[int] = None):
        """
        Initialize validator
        
        Args:
            filepath: Path to data file
            node_id_col: Column containing node IDs
            parent_id_col: Column containing parent node IDs
            max_depth: Maximum allowed hierarchy depth (None for unlimited)
        """
        self.filepath = filepath
        self.node_id_col = node_id_col
        self.parent_id_col = parent_id_col
        self.max_depth = max_depth
        self.df = None
        self.graph = defaultdict(list)  # parent -> [children]
        self.reverse_graph = defaultdict(list)  # child -> [parents]
        self.violations = []
    
    def load_data(self):
        """Load dataset"""
        if self.filepath.endswith('.csv'):
            self.df = pd.read_csv(self.filepath)
        elif self.filepath.endswith(('.xlsx', '.xls')):
            self.df = pd.read_excel(self.filepath)
        elif self.filepath.endswith('.json'):
            self.df = pd.read_json(self.filepath)
        else:
            raise ValueError("Unsupported file format")
        
        # Build graph structures
        self._build_graph()
    
    def _build_graph(self):
        """Build graph representation from data"""
        for idx, row in self.df.iterrows():
            node_id = row[self.node_id_col]
            parent_id = row[self.parent_id_col]
            
            # Skip if parent is null (root node)
            if pd.notna(parent_id):
                self.graph[parent_id].append(node_id)
                self.reverse_graph[node_id].append(parent_id)
    
    def detect_circular_references(self) -> List[Dict]:
        """Detect circular references using DFS"""
        violations = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            # Visit children
            for child in self.graph.get(node, []):
                if child not in visited:
                    result = dfs(child, path.copy())
                    if result:
                        violations.append(result)
                elif child in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(child)
                    cycle = path[cycle_start:] + [child]
                    return {
                        'type': 'circular_reference',
                        'cycle': cycle,
                        'cycle_length': len(cycle) - 1,
                        'severity': 'high'
                    }
            
            rec_stack.remove(node)
            return None
        
        # Check all nodes
        all_nodes = set(self.df[self.node_id_col].unique())
        for node in all_nodes:
            if node not in visited:
                result = dfs(node, [])
                if result:
                    violations.append(result)
        
        return violations
    
    def detect_self_references(self) -> List[Dict]:
        """Detect nodes that reference themselves as parent"""
        violations = []
        
        for idx, row in self.df.iterrows():
            node_id = row[self.node_id_col]
            parent_id = row[self.parent_id_col]
            
            if pd.notna(parent_id) and node_id == parent_id:
                violations.append({
                    'type': 'self_reference',
                    'row': int(idx),
                    'node_id': str(node_id),
                    'severity': 'high',
                    'message': f"Node {node_id} references itself as parent"
                })
        
        return violations
    
    def detect_orphaned_nodes(self) -> List[Dict]:
        """Detect nodes whose parent doesn't exist"""
        violations = []
        
        valid_node_ids = set(self.df[self.node_id_col].unique())
        
        for idx, row in self.df.iterrows():
            node_id = row[self.node_id_col]
            parent_id = row[self.parent_id_col]
            
            # Skip null parents (root nodes are valid)
            if pd.isna(parent_id):
                continue
            
            # Check if parent exists
            if parent_id not in valid_node_ids:
                violations.append({
                    'type': 'orphaned_node',
                    'row': int(idx),
                    'node_id': str(node_id),
                    'parent_id': str(parent_id),
                    'severity': 'high',
                    'message': f"Node {node_id} references non-existent parent {parent_id}"
                })
        
        return violations
    
    def detect_multiple_parents(self) -> List[Dict]:
        """Detect nodes with multiple parents (invalid for tree structure)"""
        violations = []
        
        # Count parents per node
        parent_count = defaultdict(int)
        parent_list = defaultdict(list)
        
        for idx, row in self.df.iterrows():
            node_id = row[self.node_id_col]
            parent_id = row[self.parent_id_col]
            
            if pd.notna(parent_id):
                parent_count[node_id] += 1
                parent_list[node_id].append((int(idx), str(parent_id)))
        
        # Find nodes with multiple parents
        for node_id, count in parent_count.items():
            if count > 1:
                violations.append({
                    'type': 'multiple_parents',
                    'node_id': str(node_id),
                    'parent_count': count,
                    'parents': [p[1] for p in parent_list[node_id]],
                    'rows': [p[0] for p in parent_list[node_id]],
                    'severity': 'high',
                    'message': f"Node {node_id} has {count} parent references"
                })
        
        return violations
    
    def validate_depth_constraints(self) -> List[Dict]:
        """Validate hierarchy depth constraints"""
        if self.max_depth is None:
            return []
        
        violations = []
        
        # Find root nodes (nodes with no parent)
        root_nodes = []
        for node_id in self.df[self.node_id_col].unique():
            if node_id not in self.reverse_graph or not self.reverse_graph[node_id]:
                root_nodes.append(node_id)
        
        # BFS to measure depth from each root
        for root in root_nodes:
            queue = deque([(root, 0)])  # (node, depth)
            visited = set([root])
            
            while queue:
                node, depth = queue.popleft()
                
                # Check depth constraint
                if depth > self.max_depth:
                    violations.append({
                        'type': 'depth_exceeded',
                        'node_id': str(node),
                        'depth': depth,
                        'max_depth': self.max_depth,
                        'root_node': str(root),
                        'severity': 'medium',
                        'message': f"Node {node} at depth {depth} exceeds max depth {self.max_depth}"
                    })
                
                # Visit children
                for child in self.graph.get(node, []):
                    if child not in visited:
                        visited.add(child)
                        queue.append((child, depth + 1))
        
        return violations
    
    def detect_disconnected_subgraphs(self) -> Dict:
        """Identify disconnected components in the graph"""
        all_nodes = set(self.df[self.node_id_col].unique())
        visited = set()
        components = []
        
        def bfs_component(start_node):
            component = set()
            queue = deque([start_node])
            component.add(start_node)
            
            while queue:
                node = queue.popleft()
                
                # Check children
                for child in self.graph.get(node, []):
                    if child not in component:
                        component.add(child)
                        queue.append(child)
                
                # Check parents
                for parent in self.reverse_graph.get(node, []):
                    if parent not in component:
                        component.add(parent)
                        queue.append(parent)
            
            return component
        
        # Find all connected components
        for node in all_nodes:
            if node not in visited:
                component = bfs_component(node)
                visited.update(component)
                components.append(list(component))
        
        return {
            'total_components': len(components),
            'largest_component_size': max(len(c) for c in components) if components else 0,
            'smallest_component_size': min(len(c) for c in components) if components else 0,
            'components': components if len(components) > 1 else [],
            'is_connected': len(components) == 1
        }
    
    def calculate_hierarchy_stats(self) -> Dict:
        """Calculate general statistics about the hierarchy"""
        # Find root nodes
        all_nodes = set(self.df[self.node_id_col].unique())
        nodes_with_parents = set(self.reverse_graph.keys())
        root_nodes = all_nodes - nodes_with_parents
        
        # Calculate max depth
        max_depth = 0
        for root in root_nodes:
            queue = deque([(root, 0)])
            visited = set([root])
            
            while queue:
                node, depth = queue.popleft()
                max_depth = max(max_depth, depth)
                
                for child in self.graph.get(node, []):
                    if child not in visited:
                        visited.add(child)
                        queue.append((child, depth + 1))
        
        # Count leaf nodes (nodes with no children)
        leaf_nodes = [node for node in all_nodes if not self.graph.get(node)]
        
        return {
            'total_nodes': len(all_nodes),
            'root_nodes': len(root_nodes),
            'leaf_nodes': len(leaf_nodes),
            'max_depth': max_depth,
            'avg_children_per_parent': np.mean([len(children) for children in self.graph.values()]) if self.graph else 0
        }
    
    def analyze_all(self) -> Dict:
        """Run all hierarchy validations"""
        self.load_data()
        
        results = {
            'metadata': {
                'filepath': self.filepath,
                'total_records': len(self.df),
                'node_id_column': self.node_id_col,
                'parent_id_column': self.parent_id_col
            },
            'statistics': self.calculate_hierarchy_stats(),
            'violations': {
                'self_references': self.detect_self_references(),
                'circular_references': self.detect_circular_references(),
                'orphaned_nodes': self.detect_orphaned_nodes(),
                'multiple_parents': self.detect_multiple_parents(),
                'depth_violations': self.validate_depth_constraints()
            },
            'connectivity': self.detect_disconnected_subgraphs()
        }
        
        # Calculate total violations
        total_violations = sum(
            len(v) if isinstance(v, list) else 0 
            for v in results['violations'].values()
        )
        results['metadata']['total_violations'] = total_violations
        
        return results
    
    def print_report(self, results: Dict = None):
        """Print formatted validation report"""
        if results is None:
            results = self.analyze_all()
        
        print("\n" + "="*80)
        print("HIERARCHICAL RELATIONSHIP VALIDATION REPORT")
        print("="*80)
        
        meta = results['metadata']
        print(f"\nDataset: {meta['filepath']}")
        print(f"Total Records: {meta['total_records']:,}")
        print(f"Total Violations: {meta['total_violations']:,}")
        
        # Statistics
        print("\n" + "-"*80)
        print("HIERARCHY STATISTICS")
        print("-"*80)
        stats = results['statistics']
        print(f"Total Nodes: {stats['total_nodes']:,}")
        print(f"Root Nodes: {stats['root_nodes']}")
        print(f"Leaf Nodes: {stats['leaf_nodes']}")
        print(f"Max Depth: {stats['max_depth']}")
        print(f"Avg Children per Parent: {stats['avg_children_per_parent']:.2f}")
        
        # Connectivity
        print("\n" + "-"*80)
        print("CONNECTIVITY ANALYSIS")
        print("-"*80)
        conn = results['connectivity']
        print(f"Connected: {conn['is_connected']}")
        print(f"Total Components: {conn['total_components']}")
        if not conn['is_connected']:
            print(f"Largest Component: {conn['largest_component_size']} nodes")
            print(f"Smallest Component: {conn['smallest_component_size']} nodes")
        
        # Violations
        print("\n" + "-"*80)
        print("VIOLATIONS")
        print("-"*80)
        
        violations = results['violations']
        
        # Self references
        if violations['self_references']:
            print(f"\nSelf References: {len(violations['self_references'])}")
            for v in violations['self_references'][:5]:
                print(f"  Row {v['row']}: Node {v['node_id']}")
        
        # Circular references
        if violations['circular_references']:
            print(f"\nCircular References: {len(violations['circular_references'])}")
            for v in violations['circular_references'][:5]:
                cycle_str = ' -> '.join(str(n) for n in v['cycle'])
                print(f"  Cycle (length {v['cycle_length']}): {cycle_str}")
        
        # Orphaned nodes
        if violations['orphaned_nodes']:
            print(f"\nOrphaned Nodes: {len(violations['orphaned_nodes'])}")
            for v in violations['orphaned_nodes'][:5]:
                print(f"  Row {v['row']}: Node {v['node_id']} -> missing parent {v['parent_id']}")
        
        # Multiple parents
        if violations['multiple_parents']:
            print(f"\nNodes with Multiple Parents: {len(violations['multiple_parents'])}")
            for v in violations['multiple_parents'][:5]:
                print(f"  Node {v['node_id']}: {v['parent_count']} parents")
        
        # Depth violations
        if violations['depth_violations']:
            print(f"\nDepth Constraint Violations: {len(violations['depth_violations'])}")
            for v in violations['depth_violations'][:5]:
                print(f"  Node {v['node_id']}: depth {v['depth']} > max {v['max_depth']}")
        
        print("\n" + "="*80)
    
    def export_report(self, output_path: str, results: Dict = None):
        """Export results to JSON"""
        if results is None:
            results = self.analyze_all()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Report exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate hierarchical and graph relationships'
    )
    parser.add_argument('filepath', help='Path to data file')
    parser.add_argument('--node-id', '-n', required=True,
                       help='Column containing node IDs')
    parser.add_argument('--parent-id', '-p', required=True,
                       help='Column containing parent IDs')
    parser.add_argument('--max-depth', '-d', type=int,
                       help='Maximum allowed hierarchy depth')
    parser.add_argument('--export', '-e',
                       help='Export report to JSON file')
    
    args = parser.parse_args()
    
    validator = HierarchyValidator(
        filepath=args.filepath,
        node_id_col=args.node_id,
        parent_id_col=args.parent_id,
        max_depth=args.max_depth
    )
    
    results = validator.analyze_all()
    validator.print_report(results)
    
    if args.export:
        validator.export_report(args.export, results)


if __name__ == '__main__':
    main()
  
