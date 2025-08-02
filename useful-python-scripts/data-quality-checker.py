import pandas as pd
import numpy as np

def data_quality_report(df, filename="data_quality_report.txt"):
    """
    Generate a comprehensive data quality report for any DataFrame
    """
    report = []
    report.append(f"DATA QUALITY REPORT")
    report.append("=" * 50)
    report.append(f"Dataset shape: {df.shape[0]}Rows × {df.columns} Columns\n")
    
    # Missing values analysis
    report.append("MISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    for col in df.columns:
        if missing[col] > 0:
            report.append(f"  {col}: {missing[col]} ({missing_pct[col]:.1f}%)")
    
    if missing.sum() == 0:
        report.append("  ✓ No missing values found")
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    report.append(f"\nDUPLICATE ROWS: {duplicates}")
    
    # Data types summary
    report.append(f"\nDATA TYPES:")
    for dtype in df.dtypes.value_counts().index:
        count = df.dtypes.value_counts()[dtype]
        report.append(f"  {dtype}: {count} columns")
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report.append(f"\nNUMERIC OUTLIERS (using IQR method):")
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col].count()
            if outliers > 0:
                report.append(f"  {col}: {outliers} potential outliers")
    
    # Save and print report
    report_text = "\n".join(report)
    with open(filename, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to: {filename}")

# Usage example:
# df = pd.read_csv('your_data.csv')
# data_quality_report(df)
