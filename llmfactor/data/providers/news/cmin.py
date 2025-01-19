import os
import glob
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from logging import Logger

class CMINNewsProvider:
    """Implementation of NewsProvider for CMIN format"""

    def __init__(self,
                 data_dir: str = "../../CMIN-US/news/raw",
                 logger: Optional[Logger] = None):
        """
        Initialize the NewsDataLoader with the directory containing news data files.

        Args:
            data_dir (str): Path to the directory containing the news CSV files, relative to the current file
        """
        self.data_dir = os.path.join(os.path.dirname(__file__), data_dir)
        self.news_files = sorted(glob.glob(f"{self.data_dir}/*.csv"))
        self.tickers = [file.split('/')[-1].split(".")[0] for file in self.news_files]
        self._cached_data: Dict[str, pd.DataFrame] = {}
        self.logger = logger

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

    def get_news_by_date(self,
                         ticker: str,
                         target_date: str | datetime,
                         attribute: Optional[list[str]] = None,
                         ) -> Dict[str, str]:
        """
        Get formatted news for a specific ticker and date.

        Args:
            ticker (str): Stock ticker symbol
            target_date (str): Date in format 'YYYY-MM-DD'
            attribute (list[str]): List of attributes to include in the news string
                                   allowed values: 'date', 'time', 'ticker', 'name', 'title', 'summary', 'link'

        Returns:
            str: Formatted news string
        """
        if attribute is None:
            # date time ticker name title summary link
            attribute = ['time', 'ticker', 'name', 'title', 'summary']
        if not set(attribute).issubset({'date', 'time', 'ticker', 'name', 'title', 'summary', 'link'}):
            raise ValueError("Invalid attribute")

        if type(target_date) == str:
            target_date = pd.to_datetime(target_date)

        df = self.load_news_data(ticker)
        if df is None:
            self.logger.error(f"No data available for {ticker}")
            raise ValueError(f"No data available for {ticker}")

        # Convert target_date to datetime if it's a string
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)

        # Filter news for the target date
        daily_news = df[df['date'] == target_date]

        if len(daily_news) == 0:
            self.logger.warning(f"No news available for {ticker} on {target_date}")
            raise ValueError(f"No news available for {ticker} on {target_date}")

        daily_news = daily_news[attribute]
        return daily_news.to_dict(orient='records')

    def clear_cache(self):
        """
        Clear the cached data to free up memory.
        """
        self._cached_data.clear()


if __name__ == "__main__":
    provider = CMINNewsProvider()
    print(provider.get_available_tickers())
    print(f"len: {len(provider.get_available_dates('AAPL'))}")
    news = provider.get_news_by_date('AAPL', provider.get_available_dates('AAPL')[0])
    for n in news:
        for k, v in n.items():
            print(f"{k}: {v}")
        print()