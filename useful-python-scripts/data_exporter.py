import pandas as pd
import json
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
import sqlite3
from datetime import datetime

class DataExporter:
    def __init__(self, df, base_filename=None):
        self.df = df
        self.base_filename = base_filename or f"export_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.export_log = []
    
    def to_excel_formatted(self, filename=None):
        """Export to Excel with formatting and multiple sheets"""
        filename = filename or f"{self.base_filename}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main data sheet
            self.df.to_excel(writer, sheet_name='Data', index=False)
            
            # Summary statistics sheet (for numeric data)
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary = self.df[numeric_cols].describe()
                summary.to_excel(writer, sheet_name='Summary_Stats')
            
            # Data info sheet
            info_data = {
                'Metric': ['Total Rows', 'Total Columns', 'Missing Values', 'Numeric Columns', 'Text Columns'],
                'Value': [
                    len(self.df),
                    len(self.df.columns),
                    self.df.isnull().sum().sum(),
                    len(self.df.select_dtypes(include=['number']).columns),
                    len(self.df.select_dtypes(include=['object']).columns)
                ]
            }
            info_df = pd.DataFrame(info_data)
            info_df.to_excel(writer, sheet_name='Data_Info', index=False)
            
            # Format the main data sheet
            workbook = writer.book
            worksheet = writer.sheets['Data']
            
            # Header formatting
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            
            for cell in worksheet[1]:
                cell.font = header_font
                cell.fill = header_fill
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        self.export_log.append(f"✓ Excel: {filename}")
        return filename
    
    def to_json_structured(self, filename=None):
        """Export to JSON with metadata"""
        filename = filename or f"{self.base_filename}.json"
        
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_records': len(self.df),
                'columns': list(self.df.columns),
                'data_types': {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            },
            'data': self.df.to_dict(orient='records')
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.export_log.append(f"✓ JSON: {filename}")
        return filename
    
    def to_sqlite(self, filename=None, table_name='data'):
        """Export to SQLite database"""
        filename = filename or f"{self.base_filename}.db"
        
        conn = sqlite3.connect(filename)
        
        # Export main data
        self.df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Create metadata table
        metadata = pd.DataFrame({
            'key': ['export_date', 'total_records', 'total_columns'],
            'value': [
                datetime.now().isoformat(),
                str(len(self.df)),
                str(len(self.df.columns))
            ]
        })
        metadata.to_sql('metadata', conn, if_exists='replace', index=False)
        
        conn.close()
        self.export_log.append(f"✓ SQLite: {filename}")
        return filename
    
    def to_csv_clean(self, filename=None):
        """Export to clean CSV with optimized settings"""
        filename = filename or f"{self.base_filename}.csv"
        
        # Clean data for CSV export
        df_clean = self.df.copy()
        
        # Handle potential CSV issues
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].astype(str).str.replace(',', ';').str.replace('\n', ' ')
        
        df_clean.to_csv(filename, index=False, encoding='utf-8')
        self.export_log.append(f"✓ CSV: {filename}")
        return filename
    
    def export_all(self, formats=['excel', 'json', 'csv', 'sqlite']):
        """Export to multiple formats at once"""
        print(f"Exporting data to {len(formats)} formats...")
        
        results = {}
        
        if 'excel' in formats:
            results['excel'] = self.to_excel_formatted()
        
        if 'json' in formats:
            results['json'] = self.to_json_structured()
        
        if 'csv' in formats:
            results['csv'] = self.to_csv_clean()
        
        if 'sqlite' in formats:
            results['sqlite'] = self.to_sqlite()
        
        print("\nExport Summary:")
        for log_entry in self.export_log:
            print(log_entry)
        
        return results

# Usage example:
# df = pd.read_csv('processed_data.csv')
# exporter = DataExporter(df, 'final_analysis')
# 
# # Export to all formats
# files = exporter.export_all()
# 
# # Or export to specific formats
# files = exporter.export_all(['excel', 'json'])
