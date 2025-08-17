import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BusinessMetricsAnalyzer:
    def __init__(self, data, date_col, value_col):
        self.data = data.copy()
        self.date_col = date_col
        self.value_col = value_col
        self.data[date_col] = pd.to_datetime(self.data[date_col])
        
    def calculate_growth_rate(self, period='month'):
        """Calculate growth rates with narrative explanation"""
        if period == 'month':
            grouped = self.data.groupby(self.data[self.date_col].dt.to_period('M'))[self.value_col].sum()
        else:
            grouped = self.data.groupby(self.data[self.date_col].dt.to_period('W'))[self.value_col].sum()
        
        growth_rates = grouped.pct_change() * 100
        latest_growth = growth_rates.iloc[-1]
        
        # Generate narrative
        if latest_growth > 10:
            narrative = f"Strong growth of {latest_growth:.1f}% indicates excellent performance"
        elif latest_growth > 0:
            narrative = f"Positive growth of {latest_growth:.1f}% shows steady progress"
        elif latest_growth > -5:
            narrative = f"Slight decline of {abs(latest_growth):.1f}% - monitor closely"
        else:
            narrative = f"Significant decline of {abs(latest_growth):.1f}% requires immediate attention"
            
        return {
            'growth_rate': latest_growth,
            'narrative': narrative,
            'historical_rates': growth_rates.to_dict()
        }
    
    def calculate_conversion_metrics(self, funnel_data):
        """Calculate conversion rates across funnel stages"""
        conversions = {}
        for i in range(len(funnel_data) - 1):
            stage_from = funnel_data.columns[i]
            stage_to = funnel_data.columns[i + 1]
            
            conversion_rate = (funnel_data[stage_to].sum() / funnel_data[stage_from].sum()) * 100
            conversions[f"{stage_from}_to_{stage_to}"] = conversion_rate
            
        return conversions

# Example usage
"""
# Sample data
data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=90, freq='D'),
    'revenue': np.random.normal(1000, 200, 90)
})

analyzer = BusinessMetricsAnalyzer(data, 'date', 'revenue')
growth_analysis = analyzer.calculate_growth_rate('month')
print(growth_analysis['narrative'])
