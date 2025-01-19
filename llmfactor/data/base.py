from typing import List, Protocol, Dict, Optional
from datetime import datetime
from .providers.news import CMINNewsProvider
from .providers.price import CMINPriceProvider


class NewsProvider(Protocol):
    """Protocol defining the interface for news data providers"""

    def get_news_by_date(self,
                         ticker: str,
                         target_date: str | datetime,
                         attribute: Optional[list[str]] = None,
                         ) -> List[Dict[str, str]]:
        """Get formatted news for a specific ticker and date"""
        ...

    def get_available_tickers(self) -> List[str]:
        """Get list of available tickers"""
        ...

    def get_available_dates(self, ticker: str) -> List[datetime]:
        """Get list of available dates for a ticker"""
        ...


class PriceProvider(Protocol):
    """Protocol defining the interface for price data providers"""

    def get_price_movements(self, ticker: str, target_date: datetime, window_size: int) -> List[dict]:
        """Get price movements for a window period"""
        ...

    def get_available_tickers(self) -> List[str]:
        """Get list of available tickers"""
        ...

    def get_available_dates(self, ticker: str, window_size: int) -> List[datetime]:
        """Get list of available dates for a ticker"""
        ...


class DataProviderFactory:
    """Factory for creating data providers based on configuration"""

    @staticmethod
    def create_news_provider(provider_type: str, **kwargs) -> NewsProvider:
        providers = {
            "cmin": CMINNewsProvider,
            # Add more providers here
        }
        return providers[provider_type](**kwargs)

    @staticmethod
    def create_price_provider(provider_type: str, **kwargs) -> PriceProvider:
        providers = {
            "cmin": CMINPriceProvider,
            # Add more providers here
        }
        return providers[provider_type](**kwargs)