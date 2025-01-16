import pandas as pd
import os
import glob
from typing import List, Dict, Optional
from datetime import datetime

class NewsDataLoader:
    """
    A class to load and process news data for different stock tickers.
    """
    
    def __init__(self, data_dir: str = "CMIN-US/news/raw"):
        """
        Initialize the NewsDataLoader with the directory containing news data files.
        
        Args:
            data_dir (str): Path to the directory containing the news CSV files, relative to the current file
        """
        self.data_dir = os.path.join(os.path.dirname(__file__), data_dir)
        self.news_files = sorted(glob.glob(f"{self.data_dir}/*.csv"))
        self.tickers = [file.split('/')[-1].split(".")[0] for file in self.news_files]
        self._cached_data: Dict[str, pd.DataFrame] = {}
        
    def get_available_tickers(self) -> List[str]:
        """
        Get list of available stock tickers.
        
        Returns:
            List[str]: List of available stock tickers
        """
        return self.tickers
        
    def get_available_dates(self, ticker: str) -> List[datetime]:
        """
        Get list of dates for which news is available for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            List[datetime]: Sorted list of available dates for the ticker
        """
        df = self.load_news_data(ticker)
        if df is None:
            return []
            
        # Get unique dates and sort them
        dates = sorted(df['date'].unique())
        return dates
    
    def load_news_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load news data for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing news data for the ticker,
                                  or None if ticker not found
        """
        if ticker not in self.tickers:
            print(f"Ticker {ticker} not found in available data")
            return None
            
        if ticker in self._cached_data:
            return self._cached_data[ticker]
            
        file_path = next(f for f in self.news_files if ticker in f)
        df = pd.read_csv(file_path, delimiter="\t")
        
        # Process the data
        df = df.iloc[::-1].reset_index().drop('index', axis=1)
        df['date'] = pd.to_datetime(df['date'])
        df['time'] = pd.to_datetime(df['time'])
        
        self._cached_data[ticker] = df
        return df
    
    def get_news_by_date(self, ticker: str, target_date: datetime) -> str:
        """
        Get formatted news for a specific ticker and date.
        
        Args:
            ticker (str): Stock ticker symbol
            target_date (str): Date in format 'YYYY-MM-DD'
            
        Returns:
            str: Formatted news string
        """
        df = self.load_news_data(ticker)
        if df is None:
            return "No data available for this ticker"
            
        # Convert target_date to datetime if it's a string
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
            
        # Filter news for the target date
        daily_news = df[df['date'] == target_date]
        
        if len(daily_news) == 0:
            return f"No news available for {ticker} on {target_date.strftime('%Y-%m-%d')}"
            
        # Format the news
        formatted_news = ""
        for _, row in daily_news.iterrows():
            formatted_news += f"## {row['title']}\n\n"
            formatted_news += f"{row['summary']}\n\n\n"
            
        return formatted_news

    def clear_cache(self):
        """
        Clear the cached data to free up memory.
        """
        self._cached_data.clear()


# Example usage
if __name__ == "__main__":
    # Initialize the loader
    loader = NewsDataLoader()
    
    # Get available tickers
    print("Available tickers:", loader.get_available_tickers())
    
    # Example: Get news for AAPL on a specific date
    aapl_news = loader.get_news_by_date("AAPL", datetime.strptime("2018-01-10", "%Y-%m-%d"))
    print(aapl_news)