from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import logging
from llmfactor.utils.format_text import extract_factors
from llmfactor.utils.llm_provider import LLMProvider


@dataclass
class PipelineContext:
    """Context object that gets passed through the pipeline stages"""
    ticker: str
    target_date: datetime
    raw_news: Optional[List[Dict[str, Any]]] = None
    news_text: Optional[str] = None
    price_movements: Optional[List[Dict[str, Any]]] = None
    price_history_text: Optional[str] = None
    price_target_text: Optional[str] = None
    extracted_factors: Optional[str] = None
    processed_factors: Optional[str] = None
    analysis_result: Optional[str] = None
    prediction: Optional[str] = None
    actual: Optional[str] = None
    status: str = "pending"
    error: Optional[str] = None


class PipelineStage(ABC):
    """Abstract base class for pipeline stages"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
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


class DataFetchStage(PipelineStage):
    """Fetches raw data from providers"""

    def __init__(self, news_provider, price_provider, price_k: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.news_provider = news_provider
        self.price_provider = price_provider
        self.price_k = price_k

    def process(self, context: PipelineContext) -> PipelineContext:
        context.raw_news = self.news_provider.get_news_by_date(
            context.ticker,
            context.target_date,
            attribute=['title', 'summary']
        )
        context.price_movements = self.price_provider.get_price_movements(
            context.ticker,
            context.target_date,
            self.price_k
        )
        return context


class TextFormattingStage(PipelineStage):
    """Formats raw data into text"""

    def process(self, context: PipelineContext) -> PipelineContext:
        # Format news
        news_str_format = "### {title}\n{summary}\n\n"
        context.news_text = ""
        for news in context.raw_news:
            context.news_text += news_str_format.format(
                title=news['title'],
                summary=news['summary']
            )

        # Format price movements
        price_str_format = "On {date}, the stock price of {stock_target} {risefall}.\n"
        price_str_format_last = "On {date}, the stock price of {stock_target}"

        context.price_history_text = ""
        for move in context.price_movements[:-1]:
            context.price_history_text += price_str_format.format(
                date=move['date'].strftime('%Y-%m-%d'),
                stock_target=context.ticker,
                risefall="rose" if move['rise'] else "fell"
            )

        context.price_target_text = price_str_format_last.format(
            date=context.target_date.strftime('%Y-%m-%d'),
            stock_target=context.ticker
        )
        return context


class FactorExtractionStage(PipelineStage):
    """Extracts factors using LLM"""

    def __init__(self, llm_client: LLMProvider, factor_k: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.client = llm_client
        self.factor_k = factor_k

    def process(self, context: PipelineContext) -> PipelineContext:
        response = self.client.generate_completion(
            temperature=0,
            max_tokens=self.factor_k * 200,
            messages=[
                {"role": "system",
                 "content": f"Please extract the top {self.factor_k} factors that may affect the stock price of {context.ticker} from the following news."},
                {"role": "user", "content": "# news list\n\n" + context.news_text},
                {"role": "user",
                 "content": f"Now, tell me. What are the {self.factor_k} most influential market drivers for {context.ticker} based on recent news? Write your answer with following format:\n" + "".join(
                     [f"\n{i}." for i in range(1, self.factor_k + 1)])},
            ]
        )
        context.extracted_factors = extract_factors(response.content)
        return context


class FactorPostProcessingStage(PipelineStage):
    """Post-processes extracted factors"""

    def __init__(self, llm_client: LLMProvider, method: str = "none", **kwargs):
        super().__init__(**kwargs)
        self.client = llm_client
        self.method = method

    def process(self, context: PipelineContext) -> PipelineContext:
        if self.method == "opposite":
            response = self.client.generate_completion(
                temperature=0,
                messages=[
                    {"role": "system",
                     "content": "You are a playful assistant, saying opposite situation of the given statement by user."},
                    {"role": "user", "content": context.extracted_factors}
                ]
            )
            context.processed_factors = extract_factors(response.content)
        else:
            context.processed_factors = context.extracted_factors
        return context


class PriceAnalysisStage(PipelineStage):
    """Analyzes price movements using LLM"""

    def __init__(self, llm_client: LLMProvider, **kwargs):
        super().__init__(**kwargs)
        self.client = llm_client

    def process(self, context: PipelineContext) -> PipelineContext:
        messages = [
            {"role": "system",
             "content": f"Based on the following information, please judge the direction of the {context.ticker}'s stock price from rise/fall, fill in the blank and give reasons."},
        ]

        if context.processed_factors:
            messages.append(
                {"role": "system",
                 "content": f"These are the main factors that may affect this stock's price recently:\n\n{context.processed_factors}."},
            )

        messages.extend([
            {"role": "system", "content": context.price_history_text},
            {"role": "system", "content": context.price_target_text},
        ])

        response = self.client.generate_completion(
            temperature=0,
            max_tokens=100,
            stop=["\n"],
            messages=messages
        )

        context.analysis_result = response.content
        return context


class PredictionStage(PipelineStage):
    """Extracts prediction from analysis result"""

    def process(self, context: PipelineContext) -> PipelineContext:
        filled_blanks = context.analysis_result.split('\n')[0]
        positive_sentiments = ["rise", "rose"]
        negative_sentiments = ["fall", "fell"]

        pred_rise = any(sentiment in filled_blanks for sentiment in positive_sentiments)
        pred_fall = any(sentiment in filled_blanks for sentiment in negative_sentiments)
        actual_rise = context.price_movements[-1]['rise']

        if pred_rise == pred_fall:
            context.status = "uncertain"
            return context

        context.prediction = "rise" if pred_rise else "fall"
        context.actual = "rise" if actual_rise else "fall"
        context.status = "success"
        return context


class LLMFactorPipeline:
    """Main pipeline class that orchestrates the analysis process"""

    def __init__(self):
        self.first_stage: Optional[PipelineStage] = None
        self._stages: List[PipelineStage] = []

    def add(self, stage: PipelineStage) -> 'LLMFactorPipeline':
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
            'factors': result.processed_factors,
            'analysis': result.analysis_result,
            'prediction': result.prediction,
            'actual': result.actual,
            'status': result.status,
            'error': result.error
        }
