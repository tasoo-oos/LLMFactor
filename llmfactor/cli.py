import argparse
from pathlib import Path
from .core.runner import AnalysisRunner
from .utils.logger import LoggerSingleton, ResultLogger
from llmfactor.utils.llm_provider import LLMProviderFactory
from llmfactor.data import DataProviderFactory
from llmfactor.core.pipeline import *
from llmfactor.core.scheduler import LLMFactorScheduler


def main():
    # Get the logger immediately
    logger = LoggerSingleton.get_logger()

    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='LLM Factor Analysis Tool')
        parser.add_argument('--run-name', type=str, default='test',
                            help='Name of the run (default: run)')
        parser.add_argument('--endpoint', type=str, default='http://127.0.0.1:5678/v1',
                            help='API endpoint URL (default: http://127.0.0.1:5678/v1)')
        parser.add_argument('--model', type=str, default='./models/llama-3.1-8B-instruct-Q8_0.gguf',
                            help='Model identifier to use (default: ./models/llama-3.1-8B-instruct-Q8_0.gguf)')
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
        parser.add_argument('--post-process-method', type=str, default='none', choices=['none', 'opposite'],
                            help='Post-processing method for extracted factors (default: none)')
        parser.add_argument('--max-entries', type=int, default=0,
                            help='Maximum number of entries to analyze, random sample from available tickers and dates if possible (default: 0(all))')
        args = parser.parse_args()

        # Initialize components
        llm_provider = LLMProviderFactory.create_llm_provider(
            provider_type='openai',
            base_url=args.endpoint,
            api_key=args.token,
            model=args.model,
        )
        result_logger = ResultLogger(Path(args.output), args.run_name)
        result_logger.save_settings(args)

        # Initialize providers and clients
        news_provider = DataProviderFactory.create_news_provider("cmin")
        price_provider = DataProviderFactory.create_price_provider("cmin")

        # Initialize scheduler
        scheduler = LLMFactorScheduler(price_provider, news_provider)

        # Create pipeline
        pipeline = LLMFactorPipeline()
        pipeline.add(DataFetchStage(news_provider, price_provider, price_k=args.price_k))
        pipeline.add(TextFormattingStage())
        pipeline.add(FactorExtractionStage(llm_provider, factor_k=args.factor_k))
        pipeline.add(FactorPostProcessingStage(llm_provider, method=args.post_process_method))
        pipeline.add(PriceAnalysisStage(llm_provider))
        pipeline.add(PredictionStage())

        # Create runner
        runner = AnalysisRunner(pipeline, scheduler, result_logger)

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
