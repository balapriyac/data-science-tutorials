"""
Automatically connects to data sources on schedule, pulls fresh data,
performs transformations, and saves updated datasets.
"""

import pandas as pd
import schedule
import time
from sqlalchemy import create_engine
from datetime import datetime
import logging


class DataRefresher:
    def __init__(self, log_file="data_refresh.log"):
        """Initialize the data refresher with logging."""
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def connect_database(self, connection_string):
        """Create database connection."""
        try:
            engine = create_engine(connection_string)
            self.logger.info("Database connection established")
            return engine
        except Exception as e:
            self.logger.error(f"Database connection failed: {str(e)}")
            return None
    
    def pull_data(self, engine, query, max_retries=3):
        """Pull data from database with retry logic."""
        for attempt in range(max_retries):
            try:
                df = pd.read_sql(query, engine)
                self.logger.info(f"Data pulled successfully: {len(df)} rows")
                return df
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Wait before retry
                else:
                    self.logger.error("All retry attempts failed")
                    return None
    
    def transform_data(self, df, transformations):
        """Apply data transformations."""
        try:
            for transform in transformations:
                df = transform(df)
            self.logger.info("Data transformations completed")
            return df
        except Exception as e:
            self.logger.error(f"Transformation failed: {str(e)}")
            return None
    
    def save_data(self, df, output_path, timestamp=True):
        """Save data with optional timestamp."""
        try:
            if timestamp:
                base_name = output_path.rsplit('.', 1)[0]
                extension = output_path.rsplit('.', 1)[1]
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"{base_name}_{timestamp_str}.{extension}"
            
            df.to_csv(output_path, index=False)
            self.logger.info(f"Data saved to {output_path}")
            print(f"✓ Data refreshed and saved to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Save failed: {str(e)}")
            return None
    
    def refresh_job(self, engine, query, transformations, output_path):
        """Complete refresh job."""
        self.logger.info("Starting data refresh job")
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting data refresh...")
        
        # Pull data
        df = self.pull_data(engine, query)
        if df is None:
            return
        
        # Transform data
        df = self.transform_data(df, transformations)
        if df is None:
            return
        
        # Save data
        self.save_data(df, output_path)
        
        self.logger.info("Data refresh job completed")
    
    def schedule_refresh(self, engine, query, transformations, output_path, 
                        interval_minutes=60):
        """Schedule automatic data refresh."""
        self.logger.info(f"Scheduling refresh every {interval_minutes} minutes")
        
        # Run immediately
        self.refresh_job(engine, query, transformations, output_path)
        
        # Schedule recurring job
        schedule.every(interval_minutes).minutes.do(
            self.refresh_job, engine, query, transformations, output_path
        )
        
        print(f"✓ Scheduled refresh every {interval_minutes} minutes")
        print("Press Ctrl+C to stop")
        
        # Keep running
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Scheduled refresh stopped by user")
            print("\n✓ Scheduled refresh stopped")


# Example usage
if __name__ == "__main__":
    # Example transformations
    def add_derived_metrics(df):
        """Add calculated columns."""
        if 'revenue' in df.columns and 'cost' in df.columns:
            df['profit'] = df['revenue'] - df['cost']
        return df
    
    def filter_recent(df):
        """Keep only recent records."""
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
            df = df[df['date'] >= cutoff]
        return df
    
    # Note: Replace with your actual connection details
    refresher = DataRefresher()
    
    # Example: SQLite database
    # engine = refresher.connect_database('sqlite:///sales.db')
    # query = "SELECT * FROM transactions WHERE date >= date('now', '-30 days')"
    # transformations = [add_derived_metrics, filter_recent]
    # refresher.schedule_refresh(engine, query, transformations, 
    #                           'refreshed_data.csv', interval_minutes=60)
