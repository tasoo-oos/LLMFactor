from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import argparse
from ..utils.statistics import StatisticsTracker
from ..utils.logger import ResultLogger
from .analyzer import LLMFactorAnalyzer


class AnalysisRunner:
    def __init__(self, analyzer: LLMFactorAnalyzer, logger: ResultLogger) -> None:
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

    def get_filtered_dates(self, ticker: str, args: argparse.Namespace) -> List[datetime]:
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

                    result = self.analyzer.analyze_factors(ticker, target_date, post_process_method=args.post_process_method, factor_k=args.factor_k, price_k=args.price_k)
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
