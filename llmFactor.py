import os
import time
import logging
from logging.handlers import RotatingFileHandler
from openai import OpenAI
import pandas as pd
import datetime
from dataset_manager import NewsDataLoader, PriceDataLoader
from typing import List, Dict, Any, Optional
import argparse
import json
from pathlib import Path
from tqdm import tqdm


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
    
    def get_available_dates(self, ticker: str, price_k: int) -> List[datetime.datetime]:
        """Get dates available in both price and news data for a given ticker."""
        price_dates = set(self.price_data.get_available_dates(ticker, price_k))
        news_dates = set(self.news_data.get_available_dates(ticker))
        return sorted(price_dates.intersection(news_dates))
    
    def format_price_movements(self, 
                             price_movements: List[Dict[str, Any]], 
                             stock_target: str,
                             target_date: datetime.datetime) -> tuple[str, str]:
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

    def analyze_factors(self,
                       ticker: str,
                       target_date: datetime.datetime,
                       factor_k: int = 5,
                       price_k: int = 5) -> Dict[str, Any]:
        """
        Analyze factors affecting stock price movement.
        
        Args:
            ticker: Stock ticker symbol
            target_date: Target date for analysis
            factor_k: Number of factors to extract
            price_k: Number of price movement days to consider

        Returns:
            Dictionary containing analysis results and metadata
        """

        self.logger.info(f"Analyzing factors for {ticker} on {target_date.strftime('%Y-%m-%d')}")

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

            # Extract factors
            start_time = time.time()
            factor_extraction = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=factor_k * 200,
                messages=[
                    {"role": "system", "content": f"Please extract the top {factor_k} factors that may affect the stock price of {ticker} from the following news."},
                    {"role": "user", "content": news_str}
                ]
            )
            factor_str = factor_extraction.choices[0].message.content
            result['factors'] = factor_str
            self.logger.debug(f"Factor extraction took {time.time() - start_time:.2f} seconds")

            # Analyze price movement
            start_time = time.time()
            answer_extraction = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=100,
                stop=["\n"],
                messages=[
                    {"role": "system",
                     "content": "Based on the following information, please judge the direction of the stock price from rise/fall, fill in the blank and give reasons."},
                    {"role": "user",
                     "content": f"These are the main factors that may affect this stock’s price recently: {factor_str}."},
                    {"role": "user", "content": price_str},
                    {"role": "assistant", "content": price_str_last}
                ]
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


class LoggerSetup:
    @staticmethod
    def setup(run_dir: Path) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('LLMFactorAnalyzer')
        logger.setLevel(logging.DEBUG)

        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )

        # File handler (detailed logging)
        log_file = run_dir / 'analysis.log'
        if os.path.isfile(log_file):
            os.remove(log_file)

        file_handler = RotatingFileHandler(
            run_dir / 'analysis.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        # Console handler (important info only)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


class StatisticsTracker:
    def __init__(self):
        self.stats = {
            "total": 0,
            "success": 0,
            "uncertain": 0,
            "error": 0,
            "confusion_rate": {
                "true_rise": 0,
                "true_fall": 0,
                "false_rise": 0,
                "false_fall": 0,
            }
        }

    def update(self, result: Dict[str, Any]) -> None:
        """Update statistics based on analysis result."""
        self.stats['total'] += 1

        if result['status'] == 'success':
            self.stats['success'] += 1
            if result['prediction'] == result['actual']:
                if result['prediction'] == 'rise':
                    self.stats['confusion_rate']['true_rise'] += 1
                else:
                    self.stats['confusion_rate']['true_fall'] += 1
            else:
                if result['prediction'] == 'rise':
                    self.stats['confusion_rate']['false_rise'] += 1
                else:
                    self.stats['confusion_rate']['false_fall'] += 1
        elif result['status'] == 'error':
            self.stats['error'] += 1
        elif result['status'] == 'uncertain':
            self.stats['uncertain'] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        return self.stats

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate benchmark metrics."""
        successful = self.stats['success']
        if successful == 0:
            return {
                "accuracy": float('nan'),
                "f1_score": float('nan'),
                "mcc": float('nan')
            }

        tr = self.stats['confusion_rate']['true_rise']
        tf = self.stats['confusion_rate']['true_fall']
        fr = self.stats['confusion_rate']['false_rise']
        ff = self.stats['confusion_rate']['false_fall']

        acc = (tr + tf) / successful

        f1_denominator = 2 * tr + fr + ff
        f1 = 2 * tr / f1_denominator if f1_denominator != 0 else float('nan')

        mcc_denominator = ((tr + fr) * (tr + ff) * (tf + fr) * (tf + ff)) ** 0.5
        mcc = (tr * tf - fr * ff) / mcc_denominator if mcc_denominator != 0 else float('nan')

        return {
            "accuracy": acc,
            "f1_score": f1,
            "mcc": mcc
        }

class ResultLogger:
    def __init__(self, base_dir: Path, logger: logging.Logger):
        """Initialize result logger with base directory."""
        self.logger = logger
        self.run_dir = self._create_run_directory(base_dir)
        self.failed_dir = self.run_dir / "failed"
        self.error_dir = self.failed_dir / "error"
        self.uncertain_dir = self.failed_dir / "uncertain"

        # Create directory structure
        for directory in [self.failed_dir, self.error_dir, self.uncertain_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")

    def save_settings(self, args: argparse.Namespace) -> None:
        """Save CLI arguments to settings.json."""
        settings = vars(args).copy()
        # Mask sensitive information
        if 'token' in settings:
            settings['token'] = '***masked***'

        settings_path = self.run_dir / "settings.json"
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        self.logger.info(f"Saved settings to {settings_path}")

    def _create_run_directory(self, base_dir: Path) -> Path:
        """Create timestamped run directory."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def log_result(self, result: Dict[str, Any], stats: Dict[str, Any]) -> None:
        """Log non-successful results to appropriate directory. Save statistics to stats.json too."""
        with open(self.run_dir / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        if result['status'] == 'success':
            return

        filename = f"{result['ticker']}_{result['date']}.json"
        target_dir = self.error_dir if result['status'] == 'error' else self.uncertain_dir

        with open(target_dir / filename, 'w') as f:
            json.dump(result, f, indent=2)

    def save_summary(self, stats: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """Save final summary statistics."""
        summary = {
            "statistics": stats,
            "metrics": metrics
        }
        with open(self.run_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

class AnalysisRunner:
    def __init__(self, analyzer: LLMFactorAnalyzer, logger: ResultLogger):
        self.analyzer = analyzer
        self.logger = logger
        self.stats_tracker = StatisticsTracker()

    def setup(self, args: argparse.Namespace) -> Optional[List[str]]:
        """Setup analysis run and validate inputs."""
        available_tickers = self.analyzer.get_available_tickers()
        if not available_tickers:
            print("No tickers available in both price and news data.")
            return None

        tickers_to_analyze = args.tickers if args.tickers else available_tickers
        if not set(tickers_to_analyze).issubset(available_tickers):
            print("Invalid tickers specified.")
            print(f"Invalid tickers: {set(tickers_to_analyze) - set(available_tickers)}")
            return None

        return tickers_to_analyze

    def get_filtered_dates(self, ticker: str, args: argparse.Namespace) -> List[datetime.datetime]:
        """Get filtered dates for analysis."""
        dates = self.analyzer.get_available_dates(ticker, price_k=5)

        if args.start_date:
            start_date = pd.to_datetime(args.start_date)
            dates = [d for d in dates if d >= start_date]
        if args.end_date:
            end_date = pd.to_datetime(args.end_date)
            dates = [d for d in dates if d <= end_date]

        return dates

    def calculate_total_iterations(self, tickers: List[str], args: argparse.Namespace) -> int:
        """Calculate total number of iterations for progress bar."""
        total = 0
        for ticker in tickers:
            dates = self.get_filtered_dates(ticker, args)
            total += len(dates)
        return total

    def run_analysis(self, args: argparse.Namespace) -> bool:
        """Run the complete analysis process."""
        tickers = self.setup(args)
        if not tickers:
            return False

        total_iterations = self.calculate_total_iterations(tickers, args)

        with tqdm(total=total_iterations, desc="Analyzing") as pbar:
            for ticker in tickers:
                dates = self.get_filtered_dates(ticker, args)

                for target_date in dates:
                    pbar.set_description(f"Processing {ticker} {target_date.strftime('%Y-%m-%d')}")

                    result = self.analyzer.analyze_factors(ticker, target_date)
                    self.stats_tracker.update(result)
                    self.logger.log_result(result, self.stats_tracker.get_statistics())

                    pbar.update(1)

        # Save final summary
        metrics = self.stats_tracker.calculate_metrics()
        self.logger.save_summary(self.stats_tracker.get_statistics(), metrics)
        self.print_summary(metrics)
        return True

    def print_summary(self, metrics: Dict[str, float]) -> None:
        """Print analysis summary."""
        stats = self.stats_tracker.stats

        print(f"\nAnalysis complete. Results saved to {self.logger.run_dir}")
        print(f"\nSummary:")
        print(f"Total analyses: {stats['total']}")
        print(f"Successful: {stats['success']}")
        print(f"Failed: {stats['error']}")
        print(f"Uncertain: {stats['uncertain']}")

        print(f"\nConfusion rate:")
        successful = stats['success']
        if successful > 0:
            for key, value in stats['confusion_rate'].items():
                print(f"  {key}: {value / successful:.2f}")

        print(f"\nBenchmark:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Matthews Correlation Coefficient: {metrics['mcc']:.4f}")

def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='LLM Factor Analysis Tool')
        parser.add_argument('--endpoint', type=str, default='http://localhost:11434/v1',
                            help='API endpoint URL (default: http://localhost:11434/v1)')
        parser.add_argument('--model', type=str, default='llama3.2-vision:11b',
                            help='Model identifier to use (default: llama3.2-vision:11b)')
        parser.add_argument('--token', type=str, default='token-abc123',
                            help='API token (default: token-abc123)')
        parser.add_argument('--output', type=str, default='./results',
                            help='Output directory for results (default: ./results)')
        parser.add_argument('--tickers', type=str, nargs='+',
                            help='Specific tickers to analyze (optional)')
        parser.add_argument('--start-date', type=str,
                            help='Start date for analysis (YYYY-MM-DD format)')
        parser.add_argument('--end-date', type=str,
                            help='End date for analysis (YYYY-MM-DD format)')
        parser.add_argument('--factor-k', type=int, default=5,
                            help='Number of factors to extract (default: 5)')
        parser.add_argument('--price-k', type=int, default=5,
                            help='Number of price movement days to consider (default: 5)')
        args = parser.parse_args()

        # Create result directory and set up logging
        result_dir = Path(args.output)
        os.makedirs(result_dir, exist_ok=True)

        logger = LoggerSetup.setup(result_dir)
        logger.info("Starting LLM Factor Analysis")

        # Initialize components
        analyzer = LLMFactorAnalyzer(args.endpoint, args.token, args.model, logger)
        result_logger = ResultLogger(Path(args.output), logger)

        # Save settings
        result_logger.save_settings(args)

        runner = AnalysisRunner(analyzer, result_logger)

        # Run analysis
        success = runner.run_analysis(args)
        if success:
            logger.info("Analysis completed successfully")
        else:
            logger.error("Analysis failed")

    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()