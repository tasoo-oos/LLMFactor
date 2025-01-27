from llmfactor.core.component.pipeline import PipelineStage, PipelineContext
from llmfactor.data.base import NewsProvider, PriceProvider
from llmfactor.core.component.registry import registry

@registry.register('data_fetch', 'v1')
class DataFetchStageV1(PipelineStage):
    """Fetches raw data from providers"""

    def __init__(self, news_provider: NewsProvider, price_provider: PriceProvider, price_k: int):
        super().__init__()
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
