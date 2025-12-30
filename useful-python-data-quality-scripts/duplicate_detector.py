"""
Duplicate Record Detector
Identifies exact and fuzzy duplicate records in datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
import hashlib
from difflib import SequenceMatcher

class DuplicateDetector:
    def __init__(self, filepath, key_columns=None, fuzzy_threshold=0.85):
        """
        Initialize duplicate detector
        
        Args:
            filepath: Path to data file
            key_columns: List of columns to check for duplicates (None = all columns)
            fuzzy_threshold: Similarity threshold for fuzzy matching (0-1)
        """
        self.filepath = Path(filepath)
        self.df = self._load_data()
        self.key_columns = key_columns or list(self.df.columns)
        self.fuzzy_threshold = fuzzy_threshold
        self.duplicates = []
        self.duplicate_groups = []
        self.stats = {}
        
    def _load_data(self):
        """Load data from file"""
        suffix = self.filepath.suffix.lower()
        if suffix == '.csv':
            return pd.read_csv(self.filepath)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(self.filepath)
        elif suffix == '.json':
            return pd.read_json(self.filepath)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def find_exact_duplicates(self):
        """Find exact duplicate records"""
        print("Searching for exact duplicates...")
        
        # Check for duplicates based on key columns
        subset = [col for col in self.key_columns if col in self.df.columns]
        
        # Find duplicates
        duplicated_mask = self.df.duplicated(subset=subset, keep=False)
        duplicate_df = self.df[duplicated_mask].copy()
        
        if len(duplicate_df) == 0:
            print("No exact duplicates found")
            return []
        
        # Group duplicates
        duplicate_df['_group_hash'] = duplicate_df[subset].apply(
            lambda x: hashlib.md5(str(tuple(x)).encode()).hexdigest(), axis=1
        )
        
        groups = []
        for group_hash, group in duplicate_df.groupby('_group_hash'):
            groups.append({
                'type': 'exact',
                'count': len(group),
                'rows': group.index.tolist(),
                'sample_data': group.head(1).to_dict('records')[0]
            })
        
        self.duplicate_groups.extend(groups)
        return groups
    
    def find_fuzzy_duplicates(self, columns=None):
        """
        Find fuzzy/near duplicates using string similarity
        
        Args:
            columns: Specific columns to check (default: string columns only)
        """
        print(f"Searching for fuzzy duplicates (threshold: {self.fuzzy_threshold})...")
        
        # Select string columns if not specified
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns.tolist()
            columns = [col for col in columns if col in self.key_columns]
        
        if not columns:
            print("No string columns available for fuzzy matching")
            return []
        
        fuzzy_groups = []
        processed = set()
        
        # Compare each row with others
        for i in range(len(self.df)):
            if i in processed:
                continue
            
            similar_rows = [i]
            row_i = self.df.iloc[i]
            
            for j in range(i + 1, len(self.df)):
                if j in processed:
                    continue
                
                row_j = self.df.iloc[j]
                
                # Calculate similarity across specified columns
                similarities = []
                for col in columns:
                    val_i = str(row_i[col]).lower().strip()
                    val_j = str(row_j[col]).lower().strip()
                    
                    if pd.isna(row_i[col]) and pd.isna(row_j[col]):
                        sim = 1.0
                    elif pd.isna(row_i[col]) or pd.isna(row_j[col]):
                        sim = 0.0
                    else:
                        sim = SequenceMatcher(None, val_i, val_j).ratio()
                    
                    similarities.append(sim)
                
                # Average similarity
                avg_similarity = np.mean(similarities)
                
                if avg_similarity >= self.fuzzy_threshold:
                    similar_rows.append(j)
                    processed.add(j)
            
            # If found similar rows, create a group
            if len(similar_rows) > 1:
                processed.update(similar_rows)
                group_data = self.df.iloc[similar_rows]
                
                fuzzy_groups.append({
                    'type': 'fuzzy',
                    'count': len(similar_rows),
                    'rows': similar_rows,
                    'columns_compared': columns,
                    'sample_data': group_data.head(2).to_dict('records')
                })
        
        self.duplicate_groups.extend(fuzzy_groups)
        print(f"Found {len(fuzzy_groups)} fuzzy duplicate groups")
        return fuzzy_groups
    
    def find_partial_duplicates(self, key_subset):
        """
        Find records that are duplicates on a subset of columns
        
        Args:
            key_subset: List of column names to check
        """
        print(f"Searching for partial duplicates on: {key_subset}")
        
        subset = [col for col in key_subset if col in self.df.columns]
        
        duplicated_mask = self.df.duplicated(subset=subset, keep=False)
        duplicate_df = self.df[duplicated_mask].copy()
        
        if len(duplicate_df) == 0:
            print("No partial duplicates found")
            return []
        
        duplicate_df['_group_hash'] = duplicate_df[subset].apply(
            lambda x: hashlib.md5(str(tuple(x)).encode()).hexdigest(), axis=1
        )
        
        groups = []
        for group_hash, group in duplicate_df.groupby('_group_hash'):
            groups.append({
                'type': 'partial',
                'count': len(group),
                'rows': group.index.tolist(),
                'matching_columns': subset,
                'sample_data': group.head(2).to_dict('records')
            })
        
        self.duplicate_groups.extend(groups)
        return groups
    
    def analyze_all(self, include_fuzzy=True, fuzzy_columns=None, 
                    partial_key_columns=None):
        """
        Run complete duplicate analysis
        
        Args:
            include_fuzzy: Whether to include fuzzy matching
            fuzzy_columns: Columns for fuzzy matching
            partial_key_columns: Column subsets for partial matching
        """
        print(f"Analyzing {len(self.df)} rows for duplicates...")
        
        # Reset results
        self.duplicate_groups = []
        
        # Find exact duplicates
        exact = self.find_exact_duplicates()
        
        # Find fuzzy duplicates
        if include_fuzzy:
            fuzzy = self.find_fuzzy_duplicates(fuzzy_columns)
        
        # Find partial duplicates if specified
        if partial_key_columns:
            for key_subset in partial_key_columns:
                self.find_partial_duplicates(key_subset)
        
        # Calculate statistics
        self._calculate_stats()
        
        return self.stats
    
    def _calculate_stats(self):
        """Calculate duplicate statistics"""
        total_duplicates = sum(group['count'] for group in self.duplicate_groups)
        unique_groups = len(self.duplicate_groups)
        
        # Count by type
        type_counts = defaultdict(int)
        type_records = defaultdict(int)
        for group in self.duplicate_groups:
            type_counts[group['type']] += 1
            type_records[group['type']] += group['count']
        
        # Calculate potential savings (records that could be removed)
        removable_records = sum(group['count'] - 1 for group in self.duplicate_groups)
        
        self.stats = {
            'total_rows': len(self.df),
            'total_duplicate_records': total_duplicates,
            'unique_duplicate_groups': unique_groups,
            'removable_records': removable_records,
            'duplicate_percentage': round((total_duplicates / len(self.df)) * 100, 2),
            'groups_by_type': dict(type_counts),
            'records_by_type': dict(type_records)
        }
    
    def get_deduplication_recommendations(self):
        """Generate recommendations for deduplication"""
        recommendations = []
        
        for group in self.duplicate_groups:
            if group['type'] == 'exact':
                recommendations.append({
                    'group_rows': group['rows'],
                    'action': 'safe_to_remove',
                    'keep_row': min(group['rows']),  # Keep first occurrence
                    'remove_rows': group['rows'][1:],
                    'confidence': 'high',
                    'reason': 'Exact duplicates - all fields match'
                })
            elif group['type'] == 'fuzzy':
                recommendations.append({
                    'group_rows': group['rows'],
                    'action': 'review_required',
                    'keep_row': None,
                    'remove_rows': None,
                    'confidence': 'medium',
                    'reason': 'Fuzzy match - manual review recommended'
                })
            else:  # partial
                recommendations.append({
                    'group_rows': group['rows'],
                    'action': 'review_required',
                    'keep_row': None,
                    'remove_rows': None,
                    'confidence': 'low',
                    'reason': f"Partial match on {group['matching_columns']}"
                })
        
        return recommendations
    
    def print_report(self):
        """Print duplicate analysis report"""
        print("\n" + "="*70)
        print("DUPLICATE RECORD DETECTION REPORT")
        print("="*70)
        print(f"Dataset: {self.filepath.name}")
        print(f"Total Rows: {self.stats['total_rows']:,}")
        print("="*70)
        
        if self.stats['total_duplicate_records'] == 0:
            print("\n✓ No duplicates found!")
        else:
            print(f"\nDUPLICATE SUMMARY:")
            print(f"  Total Duplicate Records: {self.stats['total_duplicate_records']:,} ({self.stats['duplicate_percentage']}%)")
            print(f"  Unique Duplicate Groups: {self.stats['unique_duplicate_groups']}")
            print(f"  Records That Could Be Removed: {self.stats['removable_records']:,}")
            
            print(f"\nBREAKDOWN BY TYPE:")
            for dtype, count in self.stats['groups_by_type'].items():
                records = self.stats['records_by_type'][dtype]
                print(f"  {dtype.capitalize()}: {count} groups ({records} records)")
            
            print(f"\nTOP DUPLICATE GROUPS:")
            sorted_groups = sorted(self.duplicate_groups, 
                                  key=lambda x: x['count'], reverse=True)
            
            for i, group in enumerate(sorted_groups[:5], 1):
                print(f"\n  Group {i} ({group['type'].upper()}):")
                print(f"    Count: {group['count']} records")
                print(f"    Rows: {group['rows'][:10]}{'...' if len(group['rows']) > 10 else ''}")
                if 'matching_columns' in group:
                    print(f"    Matching On: {group['matching_columns']}")
            
            # Recommendations
            print(f"\nRECOMMENDATIONS:")
            recs = self.get_deduplication_recommendations()
            safe_remove = sum(1 for r in recs if r['action'] == 'safe_to_remove')
            review_needed = sum(1 for r in recs if r['action'] == 'review_required')
            
            print(f"  Safe to Remove: {safe_remove} groups")
            print(f"  Manual Review Needed: {review_needed} groups")
            print(f"  Potential Space Savings: {self.stats['removable_records']} records")
        
        print("\n" + "="*70 + "\n")
    
    def export_duplicates(self, output_path='duplicates_report.json'):
        """Export duplicate analysis to JSON"""
        output = {
            'statistics': self.stats,
            'duplicate_groups': self.duplicate_groups,
            'recommendations': self.get_deduplication_recommendations()
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Duplicate analysis exported to {output_path}")
    
    def export_deduplicated(self, output_path=None, strategy='keep_first'):
        """
        Export deduplicated dataset
        
        Args:
            output_path: Output file path
            strategy: 'keep_first', 'keep_last', or 'manual'
        """
        if output_path is None:
            output_path = self.filepath.stem + '_deduplicated' + self.filepath.suffix
        
        if strategy in ['keep_first', 'keep_last']:
            # Only remove exact duplicates automatically
            rows_to_remove = set()
            for group in self.duplicate_groups:
                if group['type'] == 'exact':
                    if strategy == 'keep_first':
                        rows_to_remove.update(group['rows'][1:])
                    else:
                        rows_to_remove.update(group['rows'][:-1])
            
            clean_df = self.df.drop(index=list(rows_to_remove))
            clean_df.to_csv(output_path, index=False)
            print(f"Deduplicated dataset saved to {output_path}")
            print(f"Removed {len(rows_to_remove)} duplicate records")
        else:
            print("Manual strategy selected - export recommendations instead")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python duplicate_detector.py <filepath> [--fuzzy] [--columns col1,col2,...]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    include_fuzzy = '--fuzzy' in sys.argv
    
    # Parse column specification
    key_columns = None
    for arg in sys.argv:
        if arg.startswith('--columns='):
            key_columns = arg.split('=')[1].split(',')
    
    detector = DuplicateDetector(filepath, key_columns=key_columns)
    detector.analyze_all(include_fuzzy=include_fuzzy)
    detector.print_report()
    detector.export_duplicates()

