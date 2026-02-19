"""
Time-Series Continuity Validator
Validates temporal integrity, sequence ordering, and pattern consistency in time-series data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import argparse
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesValidator:
    """Validates time-series data for continuity, sequence, and pattern anomalies"""
    
    def __init__(self, filepath: str, timestamp_column: str, 
                 expected_frequency: Optional[str] = None,
                 velocity_rules: Optional[Dict] = None):
        """
        Initialize the validator
        
        Args:
            filepath: Path to data file
            timestamp_column: Name of timestamp column
            expected_frequency: Expected frequency ('H', 'D', 'W', 'M') or None to infer
            velocity_rules: Dict of column: max_change_per_unit rules
        """
        self.filepath = filepath
        self.timestamp_column = timestamp_column
        self.expected_frequency = expected_frequency
        self.velocity_rules = velocity_rules or {}
        self.df = None
        self.results = {}
        
    def load_data(self):
        """Load and prepare dataset"""
        if self.filepath.endswith('.csv'):
            self.df = pd.read_csv(self.filepath)
        elif self.filepath.endswith(('.xlsx', '.xls')):
            self.df = pd.read_excel(self.filepath)
        elif self.filepath.endswith('.json'):
            self.df = pd.read_json(self.filepath)
        else:
            raise ValueError("Unsupported file format")
        
        # Convert timestamp column to datetime
        self.df[self.timestamp_column] = pd.to_datetime(
            self.df[self.timestamp_column], 
            errors='coerce'
        )
        
        # Sort by timestamp
        self.df = self.df.sort_values(self.timestamp_column).reset_index(drop=True)
        
    def infer_frequency(self) -> str:
        """Infer the most likely frequency from the data"""
        if len(self.df) < 2:
            return 'unknown'
        
        time_diffs = self.df[self.timestamp_column].diff().dropna()
        median_diff = time_diffs.median()
        
        # Map to pandas frequency strings
        if median_diff < pd.Timedelta(minutes=2):
            return 'T'  # Minute
        elif median_diff < pd.Timedelta(hours=2):
            return 'H'  # Hour
        elif median_diff < pd.Timedelta(days=2):
            return 'D'  # Day
        elif median_diff < pd.Timedelta(days=8):
            return 'W'  # Week
        elif median_diff < pd.Timedelta(days=35):
            return 'M'  # Month
        else:
            return 'Y'  # Year
            
    def detect_gaps(self) -> Dict:
        """Detect missing timestamps in expected sequence"""
        if self.expected_frequency is None:
            self.expected_frequency = self.infer_frequency()
        
        if self.expected_frequency == 'unknown':
            return {'status': 'error', 'message': 'Could not infer frequency'}
        
        # Generate expected complete date range
        start_date = self.df[self.timestamp_column].min()
        end_date = self.df[self.timestamp_column].max()
        
        expected_range = pd.date_range(
            start=start_date, 
            end=end_date, 
            freq=self.expected_frequency
        )
        
        # Find missing timestamps
        actual_timestamps = set(self.df[self.timestamp_column])
        missing_timestamps = [ts for ts in expected_range if ts not in actual_timestamps]
        
        # Find gaps (consecutive missing periods)
        gaps = []
        if missing_timestamps:
            missing_timestamps.sort()
            gap_start = missing_timestamps[0]
            prev_ts = missing_timestamps[0]
            
            for ts in missing_timestamps[1:]:
                expected_next = prev_ts + pd.Timedelta(self.expected_frequency)
                if ts != expected_next:
                    # Gap ended, start new one
                    gaps.append({
                        'start': gap_start.isoformat(),
                        'end': prev_ts.isoformat(),
                        'duration': str(prev_ts - gap_start),
                        'missing_count': len([t for t in missing_timestamps 
                                            if gap_start <= t <= prev_ts])
                    })
                    gap_start = ts
                prev_ts = ts
            
            # Add final gap
            gaps.append({
                'start': gap_start.isoformat(),
                'end': prev_ts.isoformat(),
                'duration': str(prev_ts - gap_start),
                'missing_count': len([t for t in missing_timestamps 
                                    if gap_start <= t <= prev_ts])
            })
        
        return {
            'expected_frequency': self.expected_frequency,
            'expected_records': len(expected_range),
            'actual_records': len(self.df),
            'missing_records': len(missing_timestamps),
            'completeness_percentage': (len(self.df) / len(expected_range)) * 100,
            'gaps': gaps[:10],  # Top 10 gaps
            'total_gaps': len(gaps)
        }
    
    def detect_sequence_violations(self) -> Dict:
        """Detect out-of-sequence timestamps and overlaps"""
        violations = []
        
        for i in range(1, len(self.df)):
            current_ts = self.df.loc[i, self.timestamp_column]
            prev_ts = self.df.loc[i-1, self.timestamp_column]
            
            # Check for backward jumps
            if current_ts < prev_ts:
                violations.append({
                    'row': i,
                    'type': 'backward_jump',
                    'timestamp': current_ts.isoformat(),
                    'previous_timestamp': prev_ts.isoformat(),
                    'difference': str(current_ts - prev_ts)
                })
            
            # Check for duplicate timestamps
            elif current_ts == prev_ts:
                violations.append({
                    'row': i,
                    'type': 'duplicate_timestamp',
                    'timestamp': current_ts.isoformat()
                })
        
        return {
            'total_violations': len(violations),
            'backward_jumps': len([v for v in violations if v['type'] == 'backward_jump']),
            'duplicate_timestamps': len([v for v in violations if v['type'] == 'duplicate_timestamp']),
            'violations': violations[:50]  # First 50 violations
        }
    
    def check_velocity_constraints(self) -> Dict:
        """Check for impossible rate of change in values"""
        violations = []
        
        for column, max_change in self.velocity_rules.items():
            if column not in self.df.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                continue
            
            # Calculate time differences in hours
            time_diff_hours = self.df[self.timestamp_column].diff().dt.total_seconds() / 3600
            
            # Calculate value changes
            value_diff = self.df[column].diff().abs()
            
            # Calculate rate of change per hour
            rate_of_change = value_diff / time_diff_hours
            
            # Find violations
            violation_mask = rate_of_change > max_change
            violation_indices = self.df[violation_mask].index
            
            for idx in violation_indices:
                if idx > 0:
                    violations.append({
                        'row': idx,
                        'column': column,
                        'timestamp': self.df.loc[idx, self.timestamp_column].isoformat(),
                        'previous_value': float(self.df.loc[idx-1, column]),
                        'current_value': float(self.df.loc[idx, column]),
                        'change': float(value_diff.loc[idx]),
                        'time_diff_hours': float(time_diff_hours.loc[idx]),
                        'rate_of_change': float(rate_of_change.loc[idx]),
                        'max_allowed': max_change
                    })
        
        return {
            'total_violations': len(violations),
            'violations_by_column': {col: len([v for v in violations if v['column'] == col]) 
                                   for col in self.velocity_rules.keys()},
            'violations': violations[:50]
        }
    
    def detect_seasonal_violations(self) -> Dict:
        """Detect unexpected patterns (e.g., weekday data on weekends)"""
        if len(self.df) == 0:
            return {'status': 'error', 'message': 'No data to analyze'}
        
        # Add weekday information
        self.df['_weekday'] = self.df[self.timestamp_column].dt.dayofweek
        self.df['_hour'] = self.df[self.timestamp_column].dt.hour
        
        # Detect if data appears to be weekday-only
        weekday_counts = self.df['_weekday'].value_counts()
        weekend_records = weekday_counts.get(5, 0) + weekday_counts.get(6, 0)
        weekday_records = sum(weekday_counts.get(i, 0) for i in range(5))
        
        pattern_type = 'unknown'
        if weekend_records == 0 and weekday_records > 0:
            pattern_type = 'weekday_only'
        elif weekend_records > 0 and weekday_records == 0:
            pattern_type = 'weekend_only'
        elif weekend_records > 0 and weekday_records > 0:
            pattern_type = 'all_days'
        
        # Find anomalies based on pattern
        violations = []
        if pattern_type == 'weekday_only':
            weekend_mask = self.df['_weekday'].isin([5, 6])
            weekend_indices = self.df[weekend_mask].index
            violations = [{
                'row': idx,
                'type': 'unexpected_weekend_data',
                'timestamp': self.df.loc[idx, self.timestamp_column].isoformat(),
                'weekday': self.df.loc[idx, '_weekday']
            } for idx in weekend_indices[:50]]
        
        return {
            'detected_pattern': pattern_type,
            'weekday_records': int(weekday_records),
            'weekend_records': int(weekend_records),
            'total_violations': len(violations),
            'violations': violations
        }
    
    def analyze_all(self) -> Dict:
        """Run all validation checks"""
        self.load_data()
        
        self.results = {
            'metadata': {
                'filepath': self.filepath,
                'total_records': len(self.df),
                'timestamp_column': self.timestamp_column,
                'date_range': {
                    'start': self.df[self.timestamp_column].min().isoformat(),
                    'end': self.df[self.timestamp_column].max().isoformat()
                }
            },
            'gap_analysis': self.detect_gaps(),
            'sequence_analysis': self.detect_sequence_violations(),
            'velocity_analysis': self.check_velocity_constraints(),
            'seasonal_analysis': self.detect_seasonal_violations()
        }
        
        return self.results
    
    def print_report(self):
        """Print formatted validation report"""
        if not self.results:
            print("No analysis results available. Run analyze_all() first.")
            return
        
        print("\n" + "="*80)
        print("TIME-SERIES CONTINUITY VALIDATION REPORT")
        print("="*80)
        
        meta = self.results['metadata']
        print(f"\nDataset: {meta['filepath']}")
        print(f"Total Records: {meta['total_records']:,}")
        print(f"Date Range: {meta['date_range']['start']} to {meta['date_range']['end']}")
        
        # Gap Analysis
        print("\n" + "-"*80)
        print("GAP ANALYSIS")
        print("-"*80)
        gaps = self.results['gap_analysis']
        if gaps.get('completeness_percentage'):
            print(f"Completeness: {gaps['completeness_percentage']:.2f}%")
            print(f"Expected Records: {gaps['expected_records']:,}")
            print(f"Missing Records: {gaps['missing_records']:,}")
            print(f"Total Gaps: {gaps['total_gaps']}")
            if gaps['gaps']:
                print(f"\nLargest Gaps (showing up to 5):")
                for gap in sorted(gaps['gaps'], 
                                key=lambda x: x['missing_count'], 
                                reverse=True)[:5]:
                    print(f"  {gap['start']} to {gap['end']}: "
                          f"{gap['missing_count']} missing records")
        
        # Sequence Analysis
        print("\n" + "-"*80)
        print("SEQUENCE ANALYSIS")
        print("-"*80)
        seq = self.results['sequence_analysis']
        print(f"Total Violations: {seq['total_violations']}")
        print(f"Backward Jumps: {seq['backward_jumps']}")
        print(f"Duplicate Timestamps: {seq['duplicate_timestamps']}")
        if seq['violations']:
            print(f"\nSample Violations:")
            for violation in seq['violations'][:5]:
                print(f"  Row {violation['row']}: {violation['type']}")
        
        # Velocity Analysis
        print("\n" + "-"*80)
        print("VELOCITY ANALYSIS")
        print("-"*80)
        vel = self.results['velocity_analysis']
        print(f"Total Violations: {vel['total_violations']}")
        if vel['violations_by_column']:
            print(f"Violations by Column:")
            for col, count in vel['violations_by_column'].items():
                print(f"  {col}: {count}")
        
        # Seasonal Analysis
        print("\n" + "-"*80)
        print("SEASONAL PATTERN ANALYSIS")
        print("-"*80)
        seasonal = self.results['seasonal_analysis']
        print(f"Detected Pattern: {seasonal.get('detected_pattern', 'unknown')}")
        print(f"Weekday Records: {seasonal.get('weekday_records', 0):,}")
        print(f"Weekend Records: {seasonal.get('weekend_records', 0):,}")
        print(f"Pattern Violations: {seasonal.get('total_violations', 0)}")
        
        print("\n" + "="*80)
    
    def export_report(self, output_path: str):
        """Export results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Report exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate time-series data continuity and patterns'
    )
    parser.add_argument('filepath', help='Path to data file')
    parser.add_argument('--timestamp-column', '-t', required=True,
                       help='Name of timestamp column')
    parser.add_argument('--frequency', '-f', 
                       help='Expected frequency (H, D, W, M)')
    parser.add_argument('--export', '-e',
                       help='Export report to JSON file')
    
    args = parser.parse_args()
    
    validator = TimeSeriesValidator(
        filepath=args.filepath,
        timestamp_column=args.timestamp_column,
        expected_frequency=args.frequency
    )
    
    validator.analyze_all()
    validator.print_report()
    
    if args.export:
        validator.export_report(args.export)


if __name__ == '__main__':
    main()

