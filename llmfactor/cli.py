import argparse
import json
from pathlib import Path
from .core.runner import AnalysisRunner
from .util.logger import LoggerSingleton, ResultLogger
from llmfactor.util.llm_provider import LLMProviderFactory
from llmfactor.data import DataProviderFactory
from llmfactor.core.component.pipeline import *
from llmfactor.core.scheduler import LLMFactorScheduler


def main():
    # Get the logger immediately
    logger = LoggerSingleton.get_logger()

    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='LLM Factor Analysis Tool')
        parser.add_argument('--config-file', type=str, default='./config.json',
                            help='Path to the configuration file (default: ./config.json)')
        args = parser.parse_args()

        # Load configuration
        config = {}
        if Path(args.config_file).exists():
            with open(args.config_file, 'r') as f:
                config = json.load(f)

        global_config = config.get("global", {})

        # Initialize components
        llm_client = LLMProviderFactory.create_llm_provider(**global_config["llm_client"])
        result_logger = ResultLogger(**global_config["result_logger"])
        result_logger.save_settings(config)

        # Initialize providers and clients
        news_provider = DataProviderFactory.create_news_provider(global_config["data_provider"]["news"])
        price_provider = DataProviderFactory.create_price_provider(global_config["data_provider"]["price"])

        # Initialize scheduler
        scheduler = LLMFactorScheduler(price_provider, news_provider)

        dependencies = {
            "llm_client": llm_client,
            "news_provider": news_provider,
            "price_provider": price_provider,
            "price_k": global_config["else"]["price_k"],
            "factor_k": global_config["else"]["factor_k"]
        }

        # Create pipeline
        pipeline = LLMFactorPipeline(config["stages"], dependencies)

        # Create runner
        runner = AnalysisRunner(pipeline, scheduler, result_logger)

        # Run analysis
        run_settings = {**global_config["run_settings"],
                        "price_k": global_config["else"]["price_k"]}
        success = runner.run_analysis(**run_settings)
        if success:
            logger.info("Analysis completed successfully")
        else:
            logger.error("Analysis failed")

    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise e
