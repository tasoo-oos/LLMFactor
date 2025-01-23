from typing import Dict
from tqdm import tqdm
import argparse
from ..utils.statistics import StatisticsTracker
from ..utils.logger import ResultLogger
from .pipeline import LLMFactorPipeline
from .scheduler import LLMFactorScheduler
from llmfactor.utils.logger import LoggerSingleton


class AnalysisRunner:
    def __init__(self,
                 analyzer: LLMFactorPipeline,
                 scheduler: LLMFactorScheduler,
                 logger: ResultLogger,
                 ) -> None:
        self.analyzer = analyzer
        self.scheduler = scheduler
        self.result_logger = logger

        self.logger = LoggerSingleton.get_logger()
        self.stats_tracker = StatisticsTracker()

    def run_analysis(self, args: argparse.Namespace) -> bool:
        """Run the complete analysis process."""
        entries = self.scheduler.setup(args.tickers, args.start_date, args.end_date, args.price_k, args.max_entries)

        with tqdm(total=len(entries), desc="Analyzing") as pbar:
            for ticker, target_date in entries:
                pbar.set_description(f"Processing {ticker} {target_date.strftime('%Y-%m-%d')}")

                result = self.analyzer.analyze(ticker, target_date)
                self.stats_tracker.update(result)
                self.result_logger.log_result(result, self.stats_tracker.get_statistics())

                pbar.update(1)

        # Save final summary
        metrics = self.stats_tracker.calculate_metrics()
        self.result_logger.save_summary(self.stats_tracker.get_statistics(), metrics)
        self.print_summary(metrics)
        return True

    def print_summary(self, metrics: Dict[str, float]) -> None:
        """Print analysis summary."""
        stats = self.stats_tracker.stats

        print(f"\nAnalysis complete. Results saved to {self.result_logger.run_dir}")
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
