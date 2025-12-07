"""
Text Data Cleaner and Normalizer
Cleans and normalizes text data with configurable pipelines
for different column types (names, addresses, descriptions, etc.).
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from html import unescape

@dataclass
class CleaningResult:
    column: str
    transformations_applied: List[str]
    rows_modified: int
    sample_before: List[str]
    sample_after: List[str]

class TextCleaner:
    # Common abbreviation mappings
    ADDRESS_ABBREV = {
        r'\bst\.?\b': 'street', r'\bave\.?\b': 'avenue', r'\bblvd\.?\b': 'boulevard',
        r'\bdr\.?\b': 'drive', r'\bln\.?\b': 'lane', r'\brd\.?\b': 'road',
        r'\bct\.?\b': 'court', r'\bpl\.?\b': 'place', r'\bapt\.?\b': 'apartment',
        r'\bste\.?\b': 'suite', r'\bfl\.?\b': 'floor', r'\bhwy\.?\b': 'highway'
    }
    
    TITLE_ABBREV = {
        r'\bmr\.?\b': 'mr', r'\bmrs\.?\b': 'mrs', r'\bms\.?\b': 'ms',
        r'\bdr\.?\b': 'dr', r'\bprof\.?\b': 'prof', r'\bjr\.?\b': 'jr',
        r'\bsr\.?\b': 'sr'
    }
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results: List[CleaningResult] = []
    
    # === Individual Cleaning Functions ===
    
    @staticmethod
    def strip_whitespace(text: str) -> str:
        """Remove leading/trailing whitespace and normalize internal spaces."""
        if pd.isna(text):
            return text
        return ' '.join(str(text).split())
    
    @staticmethod
    def lowercase(text: str) -> str:
        """Convert to lowercase."""
        if pd.isna(text):
            return text
        return str(text).lower()
    
    @staticmethod
    def uppercase(text: str) -> str:
        """Convert to uppercase."""
        if pd.isna(text):
            return text
        return str(text).upper()
    
    @staticmethod
    def titlecase(text: str) -> str:
        """Convert to title case."""
        if pd.isna(text):
            return text
        return str(text).title()
    
    @staticmethod
    def remove_html(text: str) -> str:
        """Remove HTML tags and decode entities."""
        if pd.isna(text):
            return text
        text = str(text)
        text = unescape(text)
        text = re.sub(r'<[^>]+>', '', text)
        return text
    
    @staticmethod
    def remove_special_chars(text: str, keep_chars: str = '') -> str:
        """Remove special characters, optionally keeping specified ones."""
        if pd.isna(text):
            return text
        pattern = f'[^a-zA-Z0-9\\s{re.escape(keep_chars)}]'
        return re.sub(pattern, '', str(text))
    
    @staticmethod
    def remove_digits(text: str) -> str:
        """Remove all digits."""
        if pd.isna(text):
            return text
        return re.sub(r'\d+', '', str(text))
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text."""
        if pd.isna(text):
            return text
        return re.sub(r'https?://\S+|www\.\S+', '', str(text))
    
    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text."""
        if pd.isna(text):
            return text
        return re.sub(r'\S+@\S+\.\S+', '', str(text))
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters to ASCII equivalents."""
        if pd.isna(text):
            return text
        text = unicodedata.normalize('NFKD', str(text))
        return text.encode('ASCII', 'ignore').decode('ASCII')
    
    @staticmethod
    def remove_accents(text: str) -> str:
        """Remove accents while preserving base characters."""
        if pd.isna(text):
            return text
        nfkd = unicodedata.normalize('NFKD', str(text))
        return ''.join(c for c in nfkd if not unicodedata.combining(c))
    
    @staticmethod
    def fix_encoding(text: str) -> str:
        """Fix common encoding issues."""
        if pd.isna(text):
            return text
        replacements = {
            'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€"': '-',
            'â€"': '-', 'â€¢': '•', 'Ã©': 'é', 'Ã¨': 'è',
            '\x00': '', '\ufeff': ''
        }
        text = str(text)
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def expand_abbreviations(self, text: str, abbrev_dict: Dict[str, str]) -> str:
        """Expand abbreviations using provided dictionary."""
        if pd.isna(text):
            return text
        text = str(text).lower()
        for pattern, replacement in abbrev_dict.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    # === Preset Pipelines ===
    
    def clean_names(self, col: str) -> pd.Series:
        """Pipeline optimized for person names."""
        series = self.df[col].copy()
        original = series.copy()
        
        transformations = []
        
        series = series.apply(self.strip_whitespace)
        transformations.append('strip_whitespace')
        
        series = series.apply(self.remove_special_chars)
        transformations.append('remove_special_chars')
        
        series = series.apply(self.titlecase)
        transformations.append('titlecase')
        
        series = series.apply(self.remove_accents)
        transformations.append('remove_accents')
        
        modified = (original != series).sum()
        self._add_result(col, transformations, modified, original, series)
        
        return series
    
    def clean_addresses(self, col: str, standardize_abbrev: bool = True) -> pd.Series:
        """Pipeline optimized for addresses."""
        series = self.df[col].copy()
        original = series.copy()
        
        transformations = []
        
        series = series.apply(self.strip_whitespace)
        transformations.append('strip_whitespace')
        
        series = series.apply(self.lowercase)
        transformations.append('lowercase')
        
        if standardize_abbrev:
            series = series.apply(lambda x: self.expand_abbreviations(x, self.ADDRESS_ABBREV))
            transformations.append('expand_abbreviations')
        
        series = series.apply(self.titlecase)
        transformations.append('titlecase')
        
        modified = (original != series).sum()
        self._add_result(col, transformations, modified, original, series)
        
        return series
    
    def clean_descriptions(self, col: str) -> pd.Series:
        """Pipeline optimized for product descriptions or free text."""
        series = self.df[col].copy()
        original = series.copy()
        
        transformations = []
        
        series = series.apply(self.fix_encoding)
        transformations.append('fix_encoding')
        
        series = series.apply(self.remove_html)
        transformations.append('remove_html')
        
        series = series.apply(self.remove_urls)
        transformations.append('remove_urls')
        
        series = series.apply(self.strip_whitespace)
        transformations.append('strip_whitespace')
        
        modified = (original != series).sum()
        self._add_result(col, transformations, modified, original, series)
        
        return series
    
    def clean_codes(self, col: str) -> pd.Series:
        """Pipeline optimized for codes/IDs (uppercase, no spaces)."""
        series = self.df[col].copy()
        original = series.copy()
        
        transformations = []
        
        series = series.apply(lambda x: str(x).strip().upper().replace(' ', '') if pd.notna(x) else x)
        transformations.append('strip_uppercase_nospace')
        
        modified = (original != series).sum()
        self._add_result(col, transformations, modified, original, series)
        
        return series
    
    # === Custom Pipeline ===
    
    def clean_custom(
        self, col: str, operations: List[str], 
        custom_abbrev: Optional[Dict[str, str]] = None
    ) -> pd.Series:
        """
        Apply custom sequence of cleaning operations.
        
        Available operations: strip_whitespace, lowercase, uppercase, titlecase,
        remove_html, remove_special_chars, remove_digits, remove_urls,
        remove_emails, normalize_unicode, remove_accents, fix_encoding,
        expand_abbreviations
        """
        series = self.df[col].copy()
        original = series.copy()
        
        op_map = {
            'strip_whitespace': self.strip_whitespace,
            'lowercase': self.lowercase,
            'uppercase': self.uppercase,
            'titlecase': self.titlecase,
            'remove_html': self.remove_html,
            'remove_special_chars': self.remove_special_chars,
            'remove_digits': self.remove_digits,
            'remove_urls': self.remove_urls,
            'remove_emails': self.remove_emails,
            'normalize_unicode': self.normalize_unicode,
            'remove_accents': self.remove_accents,
            'fix_encoding': self.fix_encoding
        }
        
        applied = []
        for op in operations:
            if op == 'expand_abbreviations' and custom_abbrev:
                series = series.apply(lambda x: self.expand_abbreviations(x, custom_abbrev))
            elif op in op_map:
                series = series.apply(op_map[op])
            applied.append(op)
        
        modified = (original != series).sum()
        self._add_result(col, applied, modified, original, series)
        
        return series
    
    def clean_all(
        self, column_types: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Clean multiple columns based on their types.
        
        Args:
            column_types: Dict mapping column names to types
                         ('name', 'address', 'description', 'code')
        """
        for col, col_type in column_types.items():
            if col not in self.df.columns:
                continue
            
            if col_type == 'name':
                self.df[col] = self.clean_names(col)
            elif col_type == 'address':
                self.df[col] = self.clean_addresses(col)
            elif col_type == 'description':
                self.df[col] = self.clean_descriptions(col)
            elif col_type == 'code':
                self.df[col] = self.clean_codes(col)
        
        return self.df
    
    def _add_result(self, col, transforms, modified, original, cleaned):
        """Add cleaning result to reports."""
        sample_idx = original.dropna().head(3).index.tolist()
        self.results.append(CleaningResult(
            column=col,
            transformations_applied=transforms,
            rows_modified=modified,
            sample_before=[str(original.loc[i]) for i in sample_idx],
            sample_after=[str(cleaned.loc[i]) for i in sample_idx]
        ))
    
    def get_report(self) -> pd.DataFrame:
        """Get cleaning report as DataFrame."""
        return pd.DataFrame([{
            'column': r.column,
            'transformations': ', '.join(r.transformations_applied),
            'rows_modified': r.rows_modified,
            'sample_before': r.sample_before,
            'sample_after': r.sample_after
        } for r in self.results])


# Example usage
if __name__ == "__main__":
    # Create sample data with messy text
    df = pd.DataFrame({
        'name': ['  john SMITH  ', 'JANE   doe', 'bob wilson jr.', 'María García'],
        'address': ['123 Main St.', '456 oak AVE', '789 PINE blvd, apt. 5', '321 elm rd'],
        'description': [
            '<p>Great product!</p>',
            'Check out https://example.com for more â€" info',
            'Contact us at info@test.com  ',
            '  Multiple   spaces   here  '
        ],
        'product_code': ['abc 123', 'def-456', '  GHI789  ', 'jkl 000']
    })
    
    print("Original Data:")
    print(df)
    print("\n" + "="*60 + "\n")
    
    cleaner = TextCleaner(df)
    
    # Clean all columns by type
    cleaned_df = cleaner.clean_all({
        'name': 'name',
        'address': 'address',
        'description': 'description',
        'product_code': 'code'
    })
    
    print("Cleaned Data:")
    print(cleaned_df)
    print("\n" + "="*60 + "\n")
    
    print("Cleaning Report:")
    report = cleaner.get_report()
    for _, row in report.iterrows():
        print(f"\nColumn: {row['column']}")
        print(f"  Transformations: {row['transformations']}")
        print(f"  Rows modified: {row['rows_modified']}")
        print(f"  Before: {row['sample_before']}")
        print(f"  After: {row['sample_after']}")


