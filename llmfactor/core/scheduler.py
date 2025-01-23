import random
from typing import List, Tuple
from datetime import datetime
from llmfactor.data import NewsProvider, PriceProvider
from llmfactor.utils.logger import LoggerSingleton

class LLMFactorScheduler:
    """
    LLM Factor Analyzer Scheduler.
    Search, filter and sample (ticker, date) pair.
    """
    def __init__(self,
                 price_data: PriceProvider,
                 news_data: NewsProvider):
        """
        Initialize the LLM Factor Analyzer.

        Args:
            price_data (PriceProvider): Price Data Provider.
            news_data (NewsProvider): News Data Provider.
        """
        self.price_data = price_data
        self.news_data = news_data

        self.logger = LoggerSingleton.get_logger()

    def get_available_tickers(self) -> List[str]:
        """Get tickers available in both price and news data."""

        price_tickers = set(self.price_data.get_available_tickers())
        news_tickers = set(self.news_data.get_available_tickers())
        common_tickers = sorted(tuple(price_tickers.intersection(news_tickers)))

        if not common_tickers:
            self.logger.error("No common tickers found in price and news data.")
            raise ValueError("No common tickers found in price and news data.")

        return common_tickers

    def get_filtered_dates(self,
                           ticker: str,
                           price_k: int,
                           start_date: datetime,
                           end_date: datetime,
                           ) -> List[datetime]:
        """Search and filter dates available for a given ticker."""

        price_dates = set(self.price_data.get_available_dates(ticker, price_k))
        news_dates = set(self.news_data.get_available_dates(ticker))
        dates = sorted(price_dates.intersection(news_dates))

        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        return dates

    def setup(self,
              allowed_tickers: List[str],
              start_date: str | datetime,
              end_date: str | datetime,
              price_k: int,
              max_entries: int,
              ) -> List[Tuple[str, datetime]]:
        """
        Search, filter and sample (ticker, date) pair.

        Args:
            allowed_tickers (List[str]): Allowed tickers for analysis.
            start_date (str | datetime): Start date for analysis.
            end_date (str | datetime): End date for analysis.
            price_k (int): Needed prior days for price data.
            max_entries (int): Maximum number of entries to sample.
        """

        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        available_tickers = self.get_available_tickers()

        tickers_to_analyze = allowed_tickers if allowed_tickers else available_tickers
        if not set(tickers_to_analyze).issubset(available_tickers):
            self.logger.error("Invalid tickers provided for analysis.")
            raise ValueError("Invalid tickers provided for analysis.")

        result: List[Tuple[str, datetime]] = []

        for ticker in tickers_to_analyze:
            dates = self.get_filtered_dates(ticker, price_k=price_k, start_date=start_date, end_date=end_date)
            result.extend([(ticker, date) for date in dates])

        self.logger.info(f"Found {len(result)} entries for analysis, sampling {max_entries} entries.")

        # Set a seed for reproducibility
        random.seed(42)  # Using a constant seed for deterministic results

        # Sample from the result list
        if 0 < max_entries < len(result):
            result = random.sample(result, max_entries)

        # Sort by ticker, then date. Ensuring consistent output order.
        result = sorted(result, key=lambda x: (x[0], x[1]))

        return result
