"""
Generates interactive HTML dashboards with KPI metrics, trends,
and performance indicators.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


class DashboardGenerator:
    def __init__(self, title="Analytics Dashboard"):
        self.title = title
        self.figures = []
    
    def calculate_period_change(self, current, previous):
        """Calculate percentage change between periods."""
        if previous == 0:
            return 0
        return ((current - previous) / previous) * 100
    
    def create_kpi_card(self, metric_name, current_value, previous_value=None, format_type='number'):
        """Create a KPI metric card with trend."""
        if format_type == 'currency':
            display_value = f"${current_value:,.2f}"
        elif format_type == 'percent':
            display_value = f"{current_value:.1f}%"
        else:
            display_value = f"{current_value:,.0f}"
        
        change = None
        if previous_value is not None:
            change = self.calculate_period_change(current_value, previous_value)
        
        return {
            'name': metric_name,
            'value': display_value,
            'change': change
        }
    
    def generate_dashboard(self, df, date_col, metric_cols, output_file="dashboard.html"):
        """
        Generate complete dashboard with multiple visualizations.
        
        Args:
            df: DataFrame with time series data
            date_col: name of date column
            metric_cols: list of metric columns to visualize
            output_file: path to save HTML dashboard
        """
        # Convert date column
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Create subplots
        n_metrics = len(metric_cols)
        fig = make_subplots(
            rows=n_metrics + 1,
            cols=1,
            subplot_titles=['Key Metrics Overview'] + [f'{col} Trend' for col in metric_cols],
            vertical_spacing=0.1,
            row_heights=[0.2] + [0.8/n_metrics] * n_metrics
        )
        
        # Add KPI summary
        kpi_text = []
        for col in metric_cols:
            current = df[col].iloc[-1]
            previous = df[col].iloc[-2] if len(df) > 1 else current
            change = self.calculate_period_change(current, previous)
            
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            color = "green" if change > 0 else "red" if change < 0 else "gray"
            
            kpi_text.append(
                f"<b>{col}:</b> {current:,.0f} "
                f"<span style='color:{color}'>{arrow} {abs(change):.1f}%</span>"
            )
        
        fig.add_annotation(
            text="<br>".join(kpi_text),
            xref="paper", yref="paper",
            x=0.5, y=0.95,
            showarrow=False,
            font=dict(size=14),
            align="center"
        )
        
        # Add trend charts for each metric
        for idx, col in enumerate(metric_cols, start=2):
            fig.add_trace(
                go.Scatter(
                    x=df[date_col],
                    y=df[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(width=3),
                    marker=dict(size=8)
                ),
                row=idx, col=1
            )
            
            # Add trend line
            from numpy import polyfit, poly1d
            x_numeric = range(len(df))
            z = polyfit(x_numeric, df[col], 1)
            p = poly1d(z)
            
            fig.add_trace(
                go.Scatter(
                    x=df[date_col],
                    y=p(x_numeric),
                    mode='lines',
                    name=f'{col} Trend',
                    line=dict(dash='dash', width=2),
                    showlegend=False
                ),
                row=idx, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>{self.title}</b><br><sub>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</sub>",
                x=0.5,
                xanchor='center'
            ),
            height=300 + (n_metrics * 300),
            showlegend=True,
            template='plotly_white'
        )
        
        # Save to HTML
        fig.write_html(output_file)
        print(f"✓ Dashboard saved to {output_file}")
        
        return fig


# Example usage
if __name__ == "__main__":
    # Sample time series data
    dates = pd.date_range('2024-01-01', periods=12, freq='M')
    data = {
        'Date': dates,
        'Revenue': [100000, 105000, 110000, 108000, 115000, 120000, 
                   125000, 130000, 128000, 135000, 140000, 145000],
        'Customers': [1000, 1050, 1100, 1080, 1150, 1200,
                     1250, 1300, 1280, 1350, 1400, 1450],
        'Orders': [5000, 5200, 5500, 5300, 5700, 6000,
                  6200, 6500, 6300, 6700, 7000, 7200]
    }
    df = pd.DataFrame(data)
    
    dashboard = DashboardGenerator("Monthly Performance Dashboard")
    dashboard.generate_dashboard(
        df,
        date_col='Date',
        metric_cols=['Revenue', 'Customers', 'Orders']
    )


