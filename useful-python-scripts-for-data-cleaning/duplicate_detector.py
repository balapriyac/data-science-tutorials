"""
Duplicate Record Detector and Resolver
Identifies exact and fuzzy duplicate records and resolves them
using configurable matching and survivorship rules.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Literal, Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib

@dataclass
class DuplicateGroup:
    group_id: int
    record_indices: List[int]
    similarity_scores: List[float]
    match_type: str

class DuplicateDetector:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['_original_index'] = self.df.index
        self.duplicate_groups: List[DuplicateGroup] = []
        
    @staticmethod
    def levenshtein_similarity(s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings."""
        if pd.isna(s1) or pd.isna(s2):
            return 0.0
        return SequenceMatcher(None, str(s1).lower(), str(s2).lower()).ratio()
    
    @staticmethod
    def jaro_winkler(s1: str, s2: str) -> float:
        """Simplified Jaro-Winkler similarity."""
        if pd.isna(s1) or pd.isna(s2):
            return 0.0
        s1, s2 = str(s1).lower(), str(s2).lower()
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        match_dist = max(len1, len2) // 2 - 1
        
        s1_matches, s2_matches = [False] * len1, [False] * len2
        matches, transpositions = 0, 0
        
        for i in range(len1):
            start = max(0, i - match_dist)
            end = min(i + match_dist + 1, len2)
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = s2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        
        jaro = (matches/len1 + matches/len2 + (matches - transpositions/2)/matches) / 3
        
        prefix = 0
        for i in range(min(len1, len2, 4)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        
        return jaro + prefix * 0.1 * (1 - jaro)
    
    def find_exact_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Find exact duplicate records based on specified columns."""
        subset = subset or self.df.columns.tolist()
        subset = [c for c in subset if c != '_original_index']
        
        duplicates = self.df[self.df.duplicated(subset=subset, keep=False)]
        
        # Group duplicates
        if not duplicates.empty:
            grouped = duplicates.groupby(list(subset), dropna=False)
            for group_id, (_, group) in enumerate(grouped):
                if len(group) > 1:
                    self.duplicate_groups.append(DuplicateGroup(
                        group_id=group_id,
                        record_indices=group['_original_index'].tolist(),
                        similarity_scores=[1.0] * len(group),
                        match_type='exact'
                    ))
        
        return duplicates
    
    def find_fuzzy_duplicates(
        self,
        match_columns: List[str],
        threshold: float = 0.85,
        method: Literal['levenshtein', 'jaro_winkler'] = 'levenshtein',
        blocking_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Find fuzzy duplicate records using string similarity.
        
        Args:
            match_columns: Columns to use for fuzzy matching
            threshold: Minimum similarity score to consider a match
            method: Similarity algorithm to use
            blocking_column: Column to use for blocking (reduces comparisons)
        """
        similarity_func = self.levenshtein_similarity if method == 'levenshtein' else self.jaro_winkler
        fuzzy_matches = []
        matched_indices = set()
        
        # Create blocking groups or use entire dataset
        if blocking_column:
            blocks = self.df.groupby(blocking_column, dropna=False)
        else:
            blocks = [('all', self.df)]
        
        group_id = len(self.duplicate_groups)
        
        for _, block in blocks:
            indices = block.index.tolist()
            
            for i, idx1 in enumerate(indices):
                if idx1 in matched_indices:
                    continue
                    
                current_group = [idx1]
                current_scores = [1.0]
                
                for idx2 in indices[i+1:]:
                    if idx2 in matched_indices:
                        continue
                    
                    # Calculate combined similarity across match columns
                    similarities = []
                    for col in match_columns:
                        val1, val2 = self.df.loc[idx1, col], self.df.loc[idx2, col]
                        similarities.append(similarity_func(val1, val2))
                    
                    avg_similarity = np.mean(similarities)
                    
                    if avg_similarity >= threshold:
                        current_group.append(idx2)
                        current_scores.append(avg_similarity)
                        matched_indices.add(idx2)
                
                if len(current_group) > 1:
                    matched_indices.add(idx1)
                    self.duplicate_groups.append(DuplicateGroup(
                        group_id=group_id,
                        record_indices=[self.df.loc[i, '_original_index'] for i in current_group],
                        similarity_scores=current_scores,
                        match_type='fuzzy'
                    ))
                    fuzzy_matches.extend(current_group)
                    group_id += 1
        
        return self.df.loc[fuzzy_matches] if fuzzy_matches else pd.DataFrame()
    
    def resolve_duplicates(
        self,
        survivorship: Literal['first', 'last', 'most_complete', 'merge'] = 'most_complete',
        priority_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Resolve duplicate groups using survivorship rules.
        
        Args:
            survivorship: Rule for selecting surviving record
            priority_columns: Columns to prioritize for most_complete strategy
        """
        if not self.duplicate_groups:
            return self.df.drop(columns=['_original_index'])
        
        indices_to_drop = set()
        
        for group in self.duplicate_groups:
            group_df = self.df[self.df['_original_index'].isin(group.record_indices)]
            
            if survivorship == 'first':
                survivor_idx = group_df.index[0]
            elif survivorship == 'last':
                survivor_idx = group_df.index[-1]
            elif survivorship == 'most_complete':
                cols = priority_columns or [c for c in self.df.columns if c != '_original_index']
                completeness = group_df[cols].notna().sum(axis=1)
                survivor_idx = completeness.idxmax()
            elif survivorship == 'merge':
                survivor_idx = group_df.index[0]
                for col in self.df.columns:
                    if col == '_original_index':
                        continue
                    # Take first non-null value
                    non_null = group_df[col].dropna()
                    if not non_null.empty:
                        self.df.loc[survivor_idx, col] = non_null.iloc[0]
            
            # Mark non-survivors for removal
            for idx in group_df.index:
                if idx != survivor_idx:
                    indices_to_drop.add(idx)
        
        result = self.df.drop(index=list(indices_to_drop))
        return result.drop(columns=['_original_index'])
    
    def get_duplicate_report(self) -> pd.DataFrame:
        """Generate a report of all duplicate groups found."""
        report_data = []
        for group in self.duplicate_groups:
            for idx, score in zip(group.record_indices, group.similarity_scores):
                report_data.append({
                    'group_id': group.group_id,
                    'original_index': idx,
                    'similarity_score': round(score, 3),
                    'match_type': group.match_type
                })
        return pd.DataFrame(report_data)


# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates
    df = pd.DataFrame({
        'name': ['John Smith', 'John Smyth', 'Jane Doe', 'Jane Doe', 'Bob Wilson', 'Robert Wilson'],
        'email': ['john@email.com', 'john@email.com', 'jane@email.com', 'jane@email.com', 'bob@email.com', 'rwilson@email.com'],
        'phone': ['555-1234', '555-1234', '555-5678', '555-5678', '555-9999', '555-9999'],
        'city': ['New York', 'New York', 'Boston', 'Boston', 'Chicago', 'Chicago']
    })
    
    print("Original Data:")
    print(df)
    print("\n" + "="*60 + "\n")
    
    detector = DuplicateDetector(df)
    
    # Find exact duplicates
    print("Exact Duplicates (by email + phone):")
    exact = detector.find_exact_duplicates(subset=['email', 'phone'])
    print(exact)
    print("\n" + "="*60 + "\n")
    
    # Find fuzzy duplicates
    print("Fuzzy Duplicates (by name):")
    fuzzy = detector.find_fuzzy_duplicates(
        match_columns=['name'],
        threshold=0.75,
        method='jaro_winkler'
    )
    print(fuzzy)
    print("\n" + "="*60 + "\n")
    
    # Get duplicate report
    print("Duplicate Report:")
    print(detector.get_duplicate_report())
    print("\n" + "="*60 + "\n")
    
    # Resolve duplicates
    print("Resolved Data (keeping most complete):")
    resolved = detector.resolve_duplicates(survivorship='most_complete')
    print(resolved)


