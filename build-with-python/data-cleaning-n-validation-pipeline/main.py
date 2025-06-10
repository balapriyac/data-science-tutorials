import pandas as pd
import numpy as np
from pydantic import BaseModel, ValidationError, field_validator
from typing import Optional, List, Dict, Any

class DataValidator(BaseModel):
    """Pydantic model for data validation"""
    name: str
    age: Optional[int] = None
    email: Optional[str] = None
    salary: Optional[float] = None
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v is not None and (v < 0 or v > 120):
            raise ValueError('Age must be between 0 and 120')
        return v
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v

class DataPipeline:
    def __init__(self):
        self.cleaning_stats = {'duplicates_removed': 0, 'nulls_handled': 0, 'validation_errors': 0}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by handling duplicates and missing values"""
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        self.cleaning_stats['duplicates_removed'] = initial_rows - len(df)
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        string_columns = df.select_dtypes(include=['object']).columns
        df[string_columns] = df[string_columns].fillna('Unknown')
        
        self.cleaning_stats['nulls_handled'] = df.isnull().sum().sum()
        return df
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate each row using Pydantic model"""
        valid_rows = []
        errors = []
        
        for idx, row in df.iterrows():
            try:
                validated_row = DataValidator(**row.to_dict())
                valid_rows.append(validated_row.model_dump())
            except ValidationError as e:
                errors.append({'row': idx, 'errors': str(e)})
        
        self.cleaning_stats['validation_errors'] = len(errors)
        return pd.DataFrame(valid_rows), errors
    
    def process(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Main pipeline method"""
        cleaned_df = self.clean_data(df.copy())
        validated_df, validation_errors = self.validate_data(cleaned_df)
        
        return {
            'cleaned_data': validated_df,
            'validation_errors': validation_errors,
            'stats': self.cleaning_stats
        }


# Example usage
if __name__ == "__main__":
    # Sample messy data
    sample_data = pd.DataFrame({
    'name': ['Tara Jamison', 'Jane Smith', 'Lucy Lee', None, 'Clara Clark','Jane Smith'],
    'age': [25, -5, 25, 35, 150,-5],
    'email': ['taraj@email.com', 'invalid-email', 'lucy@email.com', 'jane@email.com', 'clara@email.com','invalid-email'],
    'salary': [50000, 60000, 50000, None, 75000,60000]
})
    
    pipeline = DataPipeline()
    result = pipeline.process(sample_data)
    
    print("Cleaned Data:")
    print(result['cleaned_data'])
    print(f"\nStats: {result['stats']}")
    print(f"Validation Errors: {len(result['validation_errors'])}")
