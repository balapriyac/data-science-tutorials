import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def quick_profile(df, save_plots=True):
    """
    Generate a quick profile of your dataset with key insights
    """
    print("DATASET PROFILE")
    print("=" * 50)
    
    # Basic info
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Column types breakdown
    print(f"\nColumn Types:")
    print(df.dtypes.value_counts().to_string())
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric Columns Summary:")
        print(df[numeric_cols].describe().round(2))
        
        # Create correlation heatmap if more than 2 numeric columns
        if len(numeric_cols) > 2 and save_plots:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig(f'correlation_matrix_{datetime.now().strftime("%Y%m%d_%H%M")}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"\nCategorical Columns:")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            
            # Show top 5 most frequent values
            if unique_count <= 20:
                top_values = df[col].value_counts().head(5)
                print(f"    Top values: {dict(top_values)}")
    
    # Missing values summary
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing Values:")
        missing_cols = missing[missing > 0]
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")
    
    # Memory optimization suggestions
    print(f"\nMemory Optimization Tips:")
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:
                print(f"  Consider converting '{col}' to category (saves memory)")
        elif df[col].dtype == 'int64':
            if df[col].min() >= 0 and df[col].max() < 255:
                print(f"  '{col}' could be uint8 instead of int64")

# Usage example:
# df = pd.read_csv('your_data.csv')
# quick_profile(df)
