from typing import List, Dict, Any
from datetime import datetime
from openai import OpenAI
import logging
import time
from dataset_manager import NewsDataLoader, PriceDataLoader


class LLMFactorAnalyzer:
    def __init__(self, base_url: str, api_key: str, model: str, logger: logging.Logger):
        """
        Initialize the LLM Factor Analyzer.

        Args:
            base_url: Base URL for the OpenAI API
            api_key: API key for authentication
            model: Model identifier to use for analysis
            logger: Logger instance for tracking operations
        """
        self.logger = logger
        self.logger.info(f"Initializing LLMFactorAnalyzer with model: {model}")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model
        self.price_data = PriceDataLoader()
        self.news_data = NewsDataLoader()

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

    def format_price_movements(self,
                               price_movements: List[Dict[str, Any]],
                               stock_target: str,
                               target_date: datetime) -> tuple[str, str]:
        """Format price movement data into a string."""
        price_str_format = "On {date}, the stock price of {stock_target} {risefall}.\n"
        price_str_format_last = "On {date}, the stock price of {stock_target}"

        price_str = ""

        for move in price_movements[:-1]:
            price_str += price_str_format.format(
                date=move['date'].strftime('%Y-%m-%d'),
                stock_target=stock_target,
                risefall="rose" if move['rise'] else "fell"
            )

        price_str_last = price_str_format_last.format(
            date=target_date.strftime('%Y-%m-%d'),
            stock_target=stock_target
        )

        return price_str, price_str_last

    def opposite_meaning_factors(self, factors: str) -> str:
        """
        Post-process factors os it has opposite meaning.

        Args:
            factors: Factors to make meaning opposite

        Returns:
            Factors with opposite meaning
        """

        factor_inverse = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system",
                 "content": f"You are a playful assistant, saying opposite meaning of the given statement by user. Your goal is to blend the opposite meaning of the given statement, so the user won't notice. Don't add markdown statements to your changes."},
                {"role": "user", "content": factors}
            ]
        )

        return factor_inverse.choices[0].message.content


    def analyze_factors(self,
                        ticker: str,
                        target_date: datetime,
                        extract_factors: bool = False,
                        post_process_method: str = "none",
                        factor_k: int = 5,
                        price_k: int = 5) -> Dict[str, Any]:
        """
        Analyze factors affecting stock price movement.

        Args:
            ticker: Stock ticker symbol
            target_date: Target date for analysis
            extract_factors: Whether to extract factors from news data
            post_process_method: Method to use for post-process factors after factor extraction, if extract_factors is False, this parameter is ignored
                none: No post-processing
                opposite: Make factors have opposite meaning
            factor_k: Number of factors to extract, if extract_factors is False, this parameter is ignored
            price_k: Number of price movement days to consider

        Returns:
            Dictionary containing analysis results and metadata
        """

        if post_process_method not in ["none", "opposite"]:
            raise ValueError(f"Invalid post-processing method: {post_process_method}")

        self.logger.info(f"Analyzing data for {ticker} on {target_date.strftime('%Y-%m-%d')}")

        result = {
            "ticker": ticker,
            "date": target_date.strftime('%Y-%m-%d'),
            "factors": None,
            "analysis": None,
            "prediction": None,
            "actual": None,
            "status": "",
            "error": ""
        }

        try:
            # Get data
            start_time = time.time()
            news_str = self.news_data.get_news_by_date(ticker, target_date)
            price_movements = self.price_data.get_price_movements(ticker, target_date, price_k)
            price_str, price_str_last = self.format_price_movements(price_movements, ticker, target_date)
            self.logger.debug(f"Data fetching took {time.time() - start_time:.2f} seconds")

            if extract_factors:
                # Extract factors
                start_time = time.time()
                factor_extraction = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=factor_k * 200,
                    messages=[
                        {"role": "system", "content": f"Please extract the top {factor_k} factors that may affect the stock price of {ticker} from the following news."},
                        {"role": "user", "content": news_str},
                        {"role": "assistant", "content": f"The following are the main {factor_k} factors that may affect {ticker}’s stock price recently:"}
                    ]
                )
                factor_str = factor_extraction.choices[0].message.content

                # Post-process factors
                if post_process_method == "opposite":
                    factor_str = self.opposite_meaning_factors(factor_str)
                elif post_process_method == "none":
                    pass
                else:
                    raise ValueError(f"Invalid post-processing method: {post_process_method}")

                result['factors'] = factor_str

                self.logger.debug(f"Factor extraction took {time.time() - start_time:.2f} seconds")
                self.logger.debug(f"{len(news_str)} characters of news data, {len(factor_str)} characters of factor data")

            answer_messages = [
                {"role": "system",
                 "content": "Based on the following information, please judge the direction of the stock price from rise/fall, fill in the blank and give reasons."}
            ]
            if extract_factors:
                answer_messages.append(
                    {"role": "user",
                     "content": f"These are the main factors that may affect this stock’s price recently: {factor_str}."},
                )
            answer_messages.extend([
                {"role": "user", "content": price_str},
                {"role": "assistant", "content": price_str_last}
            ])

            # Analyze price movement
            start_time = time.time()
            answer_extraction = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=100,
                stop=["\n"],
                messages=answer_messages
            )

            answer = answer_extraction.choices[0].message.content
            result['analysis'] = answer
            self.logger.debug(f"Price movement analysis took {time.time() - start_time:.2f} seconds")

            # Parse prediction
            filled_blanks = answer.split('\n')[0]
            positive_sentiments = ["rise", "rose"]
            negative_sentiments = ["fall", "fell"]

            pred_rise = any(sentiment in filled_blanks for sentiment in positive_sentiments)
            pred_fall = any(sentiment in filled_blanks for sentiment in negative_sentiments)
            actual_rise = price_movements[-1]['rise']

            if pred_rise == pred_fall:
                result['status'] = "uncertain"
                return result

            result['prediction'] = "rise" if pred_rise else "fall"
            result['actual'] = "rise" if actual_rise else "fall"

            result['status'] = "success"
            self.logger.info(f"Successfully analyzed {ticker} for {target_date.strftime('%Y-%m-%d')}")
            return result

        except Exception as e:
            result['status'] = "error"
            result['error'] = str(e)
            return result
