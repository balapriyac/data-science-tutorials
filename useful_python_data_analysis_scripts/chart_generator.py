"""
Generates dozens or hundreds of formatted charts from data in seconds
with consistent styling and branding.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class SmartChartGenerator:
    def __init__(self, style='default', color_palette='Set2'):
        """
        Initialize chart generator with styling preferences.
        
        Args:
            style: matplotlib style ('default', 'seaborn', 'ggplot', etc.)
            color_palette: seaborn color palette name
        """
        plt.style.use(style)
        self.colors = sns.color_palette(color_palette)
        self.default_figsize = (10, 6)
    
    def apply_branding(self, ax, title, xlabel, ylabel):
        """Apply consistent formatting to a chart."""
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=10)
    
    def generate_bar_charts(self, df, category_col, value_col, output_dir='charts'):
        """
        Generate separate bar charts for each category.
        
        Args:
            df: DataFrame containing the data
            category_col: column to split charts by
            value_col: column to plot
            output_dir: directory to save charts
        """
        Path(output_dir).mkdir(exist_ok=True)
        categories = df[category_col].unique()
        
        print(f"Generating {len(categories)} bar charts...")
        
        for category in categories:
            subset = df[df[category_col] == category]
            
            fig, ax = plt.subplots(figsize=self.default_figsize)
            
            # Create bar chart
            bars = ax.bar(range(len(subset)), subset[value_col], 
                         color=self.colors[0], alpha=0.8, edgecolor='black')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:,.0f}',
                       ha='center', va='bottom', fontsize=9)
            
            # Apply branding
            self.apply_branding(ax, 
                              f'{category} - {value_col}',
                              'Index', value_col)
            
            # Save
            filename = f"{output_dir}/{category.replace(' ', '_')}_bar.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ {len(categories)} bar charts saved to {output_dir}/")
    
    def generate_line_charts(self, df, date_col, value_col, group_col, output_dir='charts'):
        """
        Generate separate line charts for each group showing trends over time.
        
        Args:
            df: DataFrame with time series data
            date_col: date/time column
            value_col: value to plot
            group_col: column to split charts by
            output_dir: directory to save charts
        """
        Path(output_dir).mkdir(exist_ok=True)
        df[date_col] = pd.to_datetime(df[date_col])
        groups = df[group_col].unique()
        
        print(f"Generating {len(groups)} line charts...")
        
        for group in groups:
            subset = df[df[group_col] == group].sort_values(date_col)
            
            fig, ax = plt.subplots(figsize=self.default_figsize)
            
            # Create line chart
            ax.plot(subset[date_col], subset[value_col], 
                   marker='o', linewidth=2.5, markersize=8,
                   color=self.colors[1], label=group)
            
            # Add trend line
            from numpy import polyfit, poly1d
            x_numeric = range(len(subset))
            z = polyfit(x_numeric, subset[value_col], 1)
            p = poly1d(z)
            ax.plot(subset[date_col], p(x_numeric), 
                   linestyle='--', linewidth=2, alpha=0.7,
                   color=self.colors[2], label='Trend')
            
            # Apply branding
            self.apply_branding(ax,
                              f'{group} - {value_col} Over Time',
                              'Date', value_col)
            ax.legend(loc='best')
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            # Save
            filename = f"{output_dir}/{group.replace(' ', '_')}_line.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ {len(groups)} line charts saved to {output_dir}/")
    
    def generate_comparison_charts(self, df, categories, values, output_dir='charts'):
        """
        Generate comparison charts (grouped bar charts) for multiple metrics.
        
        Args:
            df: DataFrame with data
            categories: list of category columns
            values: list of value columns to compare
            output_dir: directory to save charts
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"Generating comparison charts...")
        
        for category in categories:
            unique_cats = df[category].unique()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = range(len(unique_cats))
            width = 0.8 / len(values)
            
            # Create grouped bars
            for idx, value_col in enumerate(values):
                offset = width * idx - (width * len(values) / 2) + (width / 2)
                values_data = [df[df[category] == cat][value_col].sum() 
                             for cat in unique_cats]
                
                ax.bar([pos + offset for pos in x], values_data,
                      width, label=value_col, color=self.colors[idx], 
                      alpha=0.8, edgecolor='black')
            
            # Apply branding
            ax.set_xticks(x)
            ax.set_xticklabels(unique_cats, rotation=45, ha='right')
            self.apply_branding(ax,
                              f'Comparison by {category}',
                              category, 'Values')
            ax.legend(loc='upper left', frameon=True, fancybox=True)
            
            # Save
            filename = f"{output_dir}/comparison_{category.replace(' ', '_')}.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ {len(categories)} comparison charts saved to {output_dir}/")
    
    def generate_distribution_charts(self, df, numeric_cols, output_dir='charts'):
        """
        Generate distribution histograms for numeric columns.
        
        Args:
            df: DataFrame with numeric data
            numeric_cols: list of numeric columns
            output_dir: directory to save charts
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"Generating {len(numeric_cols)} distribution charts...")
        
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=self.default_figsize)
            
            # Create histogram with KDE
            ax.hist(df[col].dropna(), bins=30, color=self.colors[3], 
                   alpha=0.7, edgecolor='black', density=True)
            
            # Add KDE curve
            from scipy import stats
            data = df[col].dropna()
            kde = stats.gaussian_kde(data)
            x_range = range(int(data.min()), int(data.max()) + 1)
            ax.plot(x_range, kde(x_range), color=self.colors[4], 
                   linewidth=3, label='KDE')
            
            # Add mean and median lines
            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {mean_val:.1f}')
            ax.axvline(median_val, color='green', linestyle='--', 
                      linewidth=2, label=f'Median: {median_val:.1f}')
            
            # Apply branding
            self.apply_branding(ax,
                              f'Distribution of {col}',
                              col, 'Density')
            ax.legend(loc='upper right')
            
            # Save
            filename = f"{output_dir}/dist_{col.replace(' ', '_')}.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ {len(numeric_cols)} distribution charts saved to {output_dir}/")
    
    def generate_all_charts(self, df, config, output_dir='charts'):
        """
        Generate all chart types based on configuration.
        
        Args:
            df: DataFrame with data
            config: dict with chart configuration
            output_dir: directory to save charts
        """
        total_charts = 0
        
        if 'bar_charts' in config:
            self.generate_bar_charts(df, 
                                    config['bar_charts']['category'],
                                    config['bar_charts']['value'],
                                    output_dir)
            total_charts += len(df[config['bar_charts']['category']].unique())
        
        if 'line_charts' in config:
            self.generate_line_charts(df,
                                     config['line_charts']['date'],
                                     config['line_charts']['value'],
                                     config['line_charts']['group'],
                                     output_dir)
            total_charts += len(df[config['line_charts']['group']].unique())
        
        if 'comparison_charts' in config:
            self.generate_comparison_charts(df,
                                           config['comparison_charts']['categories'],
                                           config['comparison_charts']['values'],
                                           output_dir)
            total_charts += len(config['comparison_charts']['categories'])
        
        if 'distribution_charts' in config:
            self.generate_distribution_charts(df,
                                             config['distribution_charts']['columns'],
                                             output_dir)
            total_charts += len(config['distribution_charts']['columns'])
        
        print(f"\n✓ Total: {total_charts} charts generated successfully!")


# Example usage
if __name__ == "__main__":
    # Sample data
    dates = pd.date_range('2024-01-01', periods=12, freq='M')
    regions = ['North', 'South', 'East', 'West']
    
    data = []
    for region in regions:
        for date in dates:
            data.append({
                'Date': date,
                'Region': region,
                'Sales': pd.np.random.randint(50000, 150000),
                'Units': pd.np.random.randint(500, 1500),
                'Profit': pd.np.random.randint(10000, 40000)
            })
    
    df = pd.DataFrame(data)
    
    # Configure chart generation
    chart_config = {
        'line_charts': {
            'date': 'Date',
            'value': 'Sales',
            'group': 'Region'
        },
        'comparison_charts': {
            'categories': ['Region'],
            'values': ['Sales', 'Units', 'Profit']
        },
        'distribution_charts': {
            'columns': ['Sales', 'Units']
        }
    }
    
    # Generate all charts
    generator = SmartChartGenerator(style='seaborn-v0_8-darkgrid', 
                                   color_palette='husl')
    generator.generate_all_charts(df, chart_config, output_dir='analysis_charts')
