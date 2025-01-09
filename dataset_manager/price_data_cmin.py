import pandas as pd
import glob
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

class PriceDataLoader:
    """
    A class to load and process price data for different stock tickers.
    """
    
    def __init__(self, data_dir: str = "./CMIN-Dataset/CMIN-US/price/raw"):
        """
        Initialize the PriceDataLoader with the directory containing price data files.
        
        Args:
            data_dir (str): Path to the directory containing the price CSV files
        """
        self.data_dir = data_dir
        self.price_files = sorted(glob.glob(f"{data_dir}/*.csv"))
        self.tickers = [file.split('/')[-1].split(".")[0] for file in self.price_files]
        self._cached_data: Dict[str, pd.DataFrame] = {}
        
    def get_available_tickers(self) -> List[str]:
        """
        Get list of available stock tickers.
        
        Returns:
            List[str]: List of available stock tickers
        """
        return self.tickers
    
    def load_price_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load price data for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing price data for the ticker,
                                  or None if ticker not found
        """
        if ticker not in self.tickers:
            print(f"Ticker {ticker} not found in available data")
            return None
            
        if ticker in self._cached_data:
            return self._cached_data[ticker]
            
        file_path = next(f for f in self.price_files if ticker in f)
        df = pd.read_csv(file_path, delimiter=",")
        
        # Process the data
        df['Date'] = pd.to_datetime(df['Date'])
        
        self._cached_data[ticker] = df
        return df
    
    def get_available_dates(self, ticker: str, window_size: int = 5) -> List[datetime]:
        """
        Get list of dates for which price data is available for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            window_size (int): Number of previous days to include
            
        Returns:
            List[datetime]: Sorted list of available dates for the ticker
        """
        df = self.load_price_data(ticker)
        if df is None:
            return []
            
        # Get unique dates and sort them
        dates = sorted(df['Date'].unique())[window_size:]
        return dates
    
    def get_price_window(self, ticker: str, target_date: str, window_size: int = 5) -> pd.DataFrame:
        """
        Get price data for a window of days up to and including the target date.
        
        Args:
            ticker (str): Stock ticker symbol
            target_date (str): Target date in format 'YYYY-MM-DD'
            window_size (int): Number of previous days to include
            
        Returns:
            pd.DataFrame: DataFrame containing price data for the window period
        """
        df = self.load_price_data(ticker)
        if df is None:
            return pd.DataFrame()
            
        # Convert target_date to datetime if it's a string
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
            
        # Get window of data
        window_data = df[df['Date'] <= target_date].tail(window_size + 1)
        return window_data
    
    def get_price_movements(self, ticker: str, target_date: str, window_size: int = 5) -> List[Dict]:
        """
        Get daily price movements (rise/fall) for a window of days up to target date.
        
        Args:
            ticker (str): Stock ticker symbol
            target_date (str): Target date in format 'YYYY-MM-DD'
            window_size (int): Number of previous days to include
            
        Returns:
            List[Dict]: List of dictionaries containing date and rise/fall boolean
        """
        window_data = self.get_price_window(ticker, target_date, window_size)
        if len(window_data) < 2:  # Need at least 2 days to calculate movement
            return []
            
        movements = []
        for i in range(1, len(window_data)):
            prev_close = window_data['Close'].iloc[i-1]
            current_close = window_data['Close'].iloc[i]
            date = window_data['Date'].iloc[i]
            
            movements.append({
                'date': date,
                'rise': bool(prev_close < current_close)
            })
            
        return movements

    def clear_cache(self):
        """
        Clear the cached data to free up memory.
        """
        self._cached_data.clear()


# Example usage
if __name__ == "__main__":
    # Initialize the loader
    loader = PriceDataLoader()
    
    # Get available tickers
    print("Available tickers:", loader.get_available_tickers())
    
    # Example: Get price movements for AAPL
    movements = loader.get_price_movements("AAPL", "2018-11-12")
    print("\nPrice movements for AAPL:")
    for movement in movements:
        direction = "up" if movement['rise'] else "down"
        print(f"{movement['date'].strftime('%Y-%m-%d')}: {direction}")