"""
Intelligently matches and reconciles records from different data sources
using fuzzy matching and flexible field parsing.
"""

import pandas as pd
from fuzzywuzzy import fuzz
from dateutil import parser
import re


class DataReconciler:
    def __init__(self, confidence_threshold=80):
        """
        Initialize the reconciler with a match confidence threshold.
        
        Args:
            confidence_threshold: minimum score (0-100) to consider a match
        """
        self.confidence_threshold = confidence_threshold
    
    def normalize_text(self, text):
        """Normalize text for better matching."""
        if pd.isna(text):
            return ""
        text = str(text).lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text
    
    def parse_flexible_date(self, date_value):
        """Parse dates from various formats."""
        if pd.isna(date_value):
            return None
        try:
            return parser.parse(str(date_value))
        except:
            return None
    
    def calculate_match_score(self, row1, row2, name_col, id_cols):
        """Calculate overall match confidence score."""
        scores = []
        
        # Name matching (fuzzy)
        if name_col:
            name1 = self.normalize_text(row1.get(name_col))
            name2 = self.normalize_text(row2.get(name_col))
            if name1 and name2:
                scores.append(fuzz.ratio(name1, name2))
        
        # ID matching (exact)
        for id_col in id_cols:
            id1 = str(row1.get(id_col, '')).strip()
            id2 = str(row2.get(id_col, '')).strip()
            if id1 and id2:
                scores.append(100 if id1 == id2 else 0)
        
        return sum(scores) / len(scores) if scores else 0
    
    def reconcile(self, df1, df2, name_col=None, id_cols=None, date_cols=None):
        """
        Reconcile two DataFrames with flexible matching.
        
        Args:
            df1, df2: DataFrames to reconcile
            name_col: column name containing names/descriptions
            id_cols: list of ID columns to match on
            date_cols: list of date columns to standardize
        
        Returns:
            DataFrame with matched records and confidence scores
        """
        if id_cols is None:
            id_cols = []
        if date_cols is None:
            date_cols = []
        
        results = []
        
        # Standardize dates
        for date_col in date_cols:
            if date_col in df1.columns:
                df1[date_col] = df1[date_col].apply(self.parse_flexible_date)
            if date_col in df2.columns:
                df2[date_col] = df2[date_col].apply(self.parse_flexible_date)
        
        # Match records
        for idx1, row1 in df1.iterrows():
            best_match = None
            best_score = 0
            
            for idx2, row2 in df2.iterrows():
                score = self.calculate_match_score(row1, row2, name_col, id_cols)
                if score > best_score:
                    best_score = score
                    best_match = row2
            
            # Create result record
            result = row1.to_dict()
            result['match_confidence'] = best_score
            result['match_status'] = 'MATCHED' if best_score >= self.confidence_threshold else 'REVIEW NEEDED'
            
            if best_match is not None and best_score >= self.confidence_threshold:
                for col in df2.columns:
                    result[f'source2_{col}'] = best_match[col]
            
            results.append(result)
        
        reconciled_df = pd.DataFrame(results)
        
        # Summary
        matched = len(reconciled_df[reconciled_df['match_status'] == 'MATCHED'])
        review = len(reconciled_df[reconciled_df['match_status'] == 'REVIEW NEEDED'])
        
        print(f"\n✓ Reconciliation complete:")
        print(f"  - Matched: {matched}")
        print(f"  - Needs review: {review}")
        
        return reconciled_df


# Example usage
if __name__ == "__main__":
    # Sample data from different sources
    crm_data = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003'],
        'customer_name': ['Acme Corp', 'TechStart Inc.', 'Global Industries'],
        'contact_date': ['2024-01-15', '01/20/2024', '2024-02-01']
    })
    
    finance_data = pd.DataFrame({
        'account_id': ['C001', 'C002', 'C004'],
        'account_name': ['ACME Corporation', 'TechStart Inc', 'DataFlow Systems'],
        'invoice_date': ['15-Jan-2024', '20-Jan-2024', '05-Feb-2024']
    })
    
    reconciler = DataReconciler(confidence_threshold=75)
    result = reconciler.reconcile(
        crm_data, finance_data,
        name_col='customer_name',
        id_cols=['customer_id'],
        date_cols=['contact_date']
    )
    
    print(result[['customer_name', 'match_confidence', 'match_status']])


