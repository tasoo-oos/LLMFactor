import argparse
from pathlib import Path
from .core.analyzer import LLMFactorAnalyzer
from .core.runner import AnalysisRunner
from .utils.logger import LoggerSingleton, ResultLogger


def main():
    # Get the logger immediately
    logger = LoggerSingleton.get_logger()

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

        # Initialize components
        analyzer = LLMFactorAnalyzer(args.endpoint, args.token, args.model, logger)
        result_logger = ResultLogger(Path(args.output))

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
