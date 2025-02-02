import inspect
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from .registry import registry
from llmfactor.util.logger import LoggerSingleton


@dataclass
class PipelineContext:
    """Context object that gets passed through the pipeline stages"""
    ticker: str
    target_date: datetime
    raw_news: Optional[List[Dict[str, Any]]] = None
    news_text: Optional[str] = None
    fake_news_text: Optional[str] = None
    price_movements: Optional[List[Dict[str, Any]]] = None
    price_history_text: Optional[str] = None
    price_target_text: Optional[str] = None
    extracted_factors: Optional[str | List[str]] = None
    analysis_result: Optional[str] = None
    prediction: Optional[str] = None
    actual: Optional[str] = None
    status: str = "pending"
    error: Optional[str] = None


class PipelineStage(ABC):
    """Abstract base class for pipeline stages"""

    def __init__(self):
        self.logger = LoggerSingleton.get_logger()
        self._next: Optional[PipelineStage] = None

    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """Process the context and modify it accordingly"""
        pass

    def next(self, stage: 'PipelineStage') -> 'PipelineStage':
        """Set the next stage in the pipeline"""
        self._next = stage
        return stage

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute this stage and pass result to next stage if it exists"""
        try:
            context = self.process(context)
            if self._next:
                return self._next.execute(context)
            return context
        except Exception as e:
            context.status = "error"
            context.error = str(e)
            self.logger.error(f"Error in {self.__class__.__name__}: {str(e)}")
            return context


class StageFactory:
    """Factory class for creating pipeline stages with dependency injection support"""

    def __init__(self, dependencies: Dict[str, Any]):
        """
        Initialize factory with external dependencies

        Args:
            dependencies: Dictionary containing external dependencies like:
                - llm_provider: LLMProvider instance
                - news_provider: NewsProvider instance
                - price_provider: PriceProvider instance
        """
        self.dependencies = dependencies

    def _get_required_dependencies(self, stage_class) -> Dict[str, Any]:
        """
        Extract required dependencies for a stage class based on its __init__ signature
        """
        signature = inspect.signature(stage_class.__init__)
        required_deps = {}

        for param_name, param in signature.parameters.items():
            if param_name in self.dependencies:
                required_deps[param_name] = self.dependencies[param_name]

        return required_deps

    def create_stage(self, stage_type: str, version: str, config: Dict[str, Any]) -> PipelineStage:
        """
        Create a pipeline stage with injected dependencies

        Args:
            stage_type: Type of stage (e.g., 'data_fetch', 'factor_extract')
            version: Version of the stage implementation
            config: Stage-specific configuration parameters
        """

        # Get stage class from registry
        stage_class = registry.get(stage_type, version)

        # Get required dependencies for the stage class
        required_deps = self._get_required_dependencies(stage_class)

        # Merge stage config with required dependencies
        init_params = {**config, **required_deps}

        return stage_class(**init_params)


class LLMFactorPipeline:
    """Main pipeline class that orchestrates the analysis process"""

    def __init__(self, config: List[Dict[str, Any]], dependencies: Dict[str, Any]):
        """
        Initialize pipeline from configuration

        Args:
            config: Configuration dictionary containing:
                - stages: List of stage configurations
                - global: Global configuration parameters
        """
        self.config = config
        self.factory = StageFactory(dependencies)
        self.first_stage: Optional[PipelineStage] = None
        self._stages: List[PipelineStage] = []
        self._build_pipeline()

    def _build_pipeline(self):
        """Build pipeline from configuration"""
        for stage_config in self.config:
            stage = self.factory.create_stage(
                stage_type=stage_config["type"],
                version=stage_config["version"],
                config=stage_config.get("config", {}),
            )
            self._add(stage)

    def _add(self, stage: PipelineStage) -> 'LLMFactorPipeline':
        """Add a new stage to the pipeline"""
        if not self.first_stage:
            self.first_stage = stage
        elif self._stages:
            self._stages[-1].next(stage)
        self._stages.append(stage)
        return self

    def analyze(self, ticker: str, target_date: datetime) -> Dict[str, Any]:
        """Execute the pipeline"""
        if not self.first_stage:
            raise ValueError("Pipeline has no stages")

        context = PipelineContext(ticker=ticker, target_date=target_date)
        result = self.first_stage.execute(context)

        return {
            'ticker': result.ticker,
            'date': result.target_date.strftime('%Y-%m-%d'),
            'news': result.news_text,
            'fake_news': result.fake_news_text,
            'factors': context.extracted_factors,
            'analysis': result.analysis_result,
            'prediction': result.prediction,
            'actual': result.actual,
            'status': result.status,
            'error': result.error
        }
