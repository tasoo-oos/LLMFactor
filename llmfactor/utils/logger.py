import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any
import argparse
import json
import time
import sys


class LoggerSingleton:
    _instance = None
    _initialized = False

    @classmethod
    def get_logger(cls, log_dir: Path = None) -> logging.Logger:
        if not cls._instance:
            cls._instance = logging.getLogger('LLMFactorAnalyzer')

            if not cls._initialized:
                # Set up basic console logging immediately
                cls._setup_basic_logging()
                cls._initialized = True

            # If log_dir is provided, add file handling
            if log_dir:
                cls._setup_file_logging(log_dir)

        return cls._instance

    @classmethod
    def _setup_basic_logging(cls):
        """Set up basic console logging for startup"""
        logger = cls._instance
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        logger.addHandler(console_handler)

    @classmethod
    def _setup_file_logging(cls, log_dir: Path):
        """Add file logging once we have a directory"""
        logger = cls._instance

        file_handler = RotatingFileHandler(
            log_dir / 'analysis.log',
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)


class ResultLogger:
    def __init__(self, base_dir: Path):
        """Initialize result logger with base directory."""
        self.run_dir = self._create_run_directory(base_dir)
        self.failed_dir = self.run_dir / "failed"
        self.error_dir = self.failed_dir / "error"
        self.uncertain_dir = self.failed_dir / "uncertain"

        # Use the singleton logger and add file logging
        self.logger = LoggerSingleton.get_logger(self.run_dir)

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
