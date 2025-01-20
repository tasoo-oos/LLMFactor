from typing import List, Dict, Any, Optional
from datetime import datetime
from llmfactor.utils.llm_provider import LLMProvider
import logging
import time
import re
from llmfactor.data import DataProviderFactory
from dataclasses import dataclass


@dataclass
class AnalysisResult:
    ticker: str
    date: str
    factors: Optional[str] = None
    analysis: Optional[str] = None
    prediction: Optional[str] = None
    actual: Optional[str] = None
    status: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'date': self.date,
            'factors': self.factors,
            'analysis': self.analysis,
            'prediction': self.prediction,
            'actual': self.actual,
            'status': self.status,
            'error': self.error
        }


class LLMFactorAnalyzer:
    def __init__(self, llm_provider: LLMProvider, logger: logging.Logger):
        """
        Initialize the LLM Factor Analyzer.

        Args:
            logger: Logger instance for tracking operations
        """
        self.logger = logger
        self.provider = llm_provider

        self.price_data = DataProviderFactory.create_price_provider("cmin")
        self.news_data = DataProviderFactory.create_news_provider("cmin")

    def get_available_tickers(self) -> List[str]:
        """Get tickers available in both price and news data."""
        price_tickers = set(self.price_data.get_available_tickers())
        news_tickers = set(self.news_data.get_available_tickers())
        common_tickers = sorted(tuple(price_tickers.intersection(news_tickers)))
        self.logger.debug(f"Found {len(common_tickers)} common tickers")
        return common_tickers

    def get_available_dates(self, ticker: str, price_k: int) -> List[datetime]:
        """Get dates available in both price and news data for a given ticker."""
        price_dates = set(self.price_data.get_available_dates(ticker, price_k))
        news_dates = set(self.news_data.get_available_dates(ticker))
        return sorted(price_dates.intersection(news_dates))

    def format_news_data(self, news_data: List[Dict[str, Any]]) -> str:
        """Format news data into a string."""
        news_str_format = "### {title}\n{summary}\n\n"
        news_str = ""

        for news in news_data:
            news_str += news_str_format.format(
                title=news['title'],
                summary=news['summary']
            )

        return news_str

    def format_price_movements(self,
                               price_movements: List[Dict[str, Any]],
                               stock_target: str,
                               target_date: datetime) -> str:
        """Format price movement data into a string."""
        price_str_format = "On {date}, the stock price of {stock_target} {risefall}.\n"
        price_str_format_last = "On {date}, the stock price of {stock_target} ____."

        price_str = ""

        for move in price_movements[:-1]:
            price_str += price_str_format.format(
                date=move['date'].strftime('%Y-%m-%d'),
                stock_target=stock_target,
                risefall="rose" if move['rise'] else "fell"
            )

        price_str += price_str_format_last.format(
            date=target_date.strftime('%Y-%m-%d'),
            stock_target=stock_target
        )

        return price_str

    def extract_factors(self, text: str) -> str:
        """Extract factors from text using regex."""
        # Pattern to match numbered items with their descriptions
        # Looks for: number, dot, colon, and remaining text
        pattern = r'\d+\.[^:]+:\s+([^\n]+)'

        # Find all matches in the text
        matches = re.finditer(pattern, text)

        # Create a dictionary to store factors with their descriptions
        result_str = ""

        for match in matches:
            result_str += match.group(0) + "\n"

        return result_str[:-1]

    def _fetch_analysis_data(self,
                             ticker: str,
                             target_date: datetime,
                             price_k: int) -> Dict[str, Any]:
        """Fetch and prepare all necessary data for analysis."""
        start_time = time.time()

        news_data = self.news_data.get_news_by_date(ticker, target_date, attribute=['title', 'summary'])
        news_str = self.format_news_data(news_data)
        price_movements = self.price_data.get_price_movements(ticker, target_date, price_k)
        price_str = self.format_price_movements(price_movements, ticker, target_date)

        self.logger.debug(f"Data fetching took {time.time() - start_time:.2f} seconds")

        return {
            'news': news_str,
            'price_movements': price_movements,
            'price_str': price_str
        }

    def _process_factors(self,
                         ticker: str,
                         news_str: str,
                         factor_k: int,
                         processing_method: str) -> str:
        """Extract and post-process factors from news data."""
        start_time = time.time()

        if processing_method == "opposite-factors":
            factor_str = self._extract_factors(ticker, news_str, factor_k)
            factor_str = self.opposite_meaning_factors(factor_str)
        elif processing_method == "none":
            factor_str = self._extract_factors(ticker, news_str, factor_k)
        else:
            raise ValueError(f"Invalid post-processing method: {processing_method}")

        self.logger.debug(f"Factor processing took {time.time() - start_time:.2f} seconds")
        return factor_str

    def _extract_factors(self,
                         ticker: str,
                         news_str: str,
                         factor_k: int) -> str:
        """Extract factors using LLM."""
        response = self.provider.generate_completion(
            temperature=0,
            max_tokens=factor_k * 200,
            messages=[
                {"role": "system",
                 "content": f"Please extract the top {factor_k} factors that may affect the stock price of {ticker} from the following news."},
                {"role": "user", "content": "# news list\n\n" + news_str},
                {"role": "user",
                 "content": f"Now, tell me. What are the {factor_k} most influential market drivers for {ticker} based on recent news? Write your answer with following format:\n" + "".join(
                     [f"\n{i}." for i in range(1, factor_k + 1)])},
            ]
        )

        return self.extract_factors(response.content)

    def opposite_meaning_factors(self, factors: str) -> str:
        """
        Post-process factors os it has opposite meaning.

        Args:
            factors: Factors to make meaning opposite

        Returns:
            Factors with opposite meaning
        """

        response = self.provider.generate_completion(
            temperature=0,
            messages=[
                {"role": "system",
                 "content": f"You are a playful assistant, saying opposite situation of the given statement by user. Your goal is to write the opposite situation of each given statement."},
                {"role": "user", "content": factors}
            ]
        )

        return self.extract_factors(response.content)

    def _analyze_price_movement(self,
                                ticker: str,
                                news_factors: Optional[str],
                                price_str: str) -> str:
        """Analyze price movement using LLM."""
        start_time = time.time()


        response = self.provider.generate_completion(
            temperature=0,
            max_tokens=100,
            stop=["\n"],
            messages=[
            {"role": "system",
             "content": f"Based on the following information, please judge the direction of the {ticker}'s stock price from rise/fall, fill in the blank and give reasons."},
            {"role": "system",
             "content": f"These are the main factors that may affect this stock’s price recently:\n\n{news_factors}."},
            {"role": "system", "content": price_str}]
        )

        self.logger.debug(f"Price movement analysis took {time.time() - start_time:.2f} seconds")
        return response.content

    def _update_result_with_prediction(self,
                                       result: AnalysisResult,
                                       analysis: str,
                                       price_movements: List[Dict[str, Any]]) -> None:
        """Update result object with prediction analysis."""
        result.analysis = analysis

        filled_blanks = analysis.split('\n')[0]
        positive_sentiments = ["rise", "rose"]
        negative_sentiments = ["fall", "fell"]

        pred_rise = any(sentiment in filled_blanks for sentiment in positive_sentiments)
        pred_fall = any(sentiment in filled_blanks for sentiment in negative_sentiments)
        actual_rise = price_movements[-1]['rise']

        if pred_rise == pred_fall:
            result.status = "uncertain"
            return

        result.prediction = "rise" if pred_rise else "fall"
        result.actual = "rise" if actual_rise else "fall"

    def analyze_factors(self,
                        ticker: str,
                        target_date: datetime,
                        processing_method: str = "none",
                        factor_k: int = 5,
                        price_k: int = 5) -> Dict[str, Any]:
        """
        Main entry point for factor analysis. Orchestrates the analysis workflow.
        """
        self.logger.info(f"Analyzing data for {ticker} on {target_date.strftime('%Y-%m-%d')}")

        result = AnalysisResult(
            ticker=ticker,
            date=target_date.strftime('%Y-%m-%d')
        )

        try:
            data = self._fetch_analysis_data(ticker, target_date, price_k)

            result.factors = self._process_factors(
                ticker,
                data['news'],
                factor_k,
                processing_method
            )

            analysis_result = self._analyze_price_movement(
                ticker=ticker,
                news_factors=result.factors,
                price_str=data['price_str']
            )

            self._update_result_with_prediction(result, analysis_result, data['price_movements'])

            result.status = "success"
            self.logger.info(f"Successfully analyzed {ticker} for {target_date.strftime('%Y-%m-%d')}")

        except Exception as e:
            result.status = "error"
            result.error = str(e)
            self.logger.error(f"Error analyzing {ticker}: {str(e)}")

        return result.to_dict()