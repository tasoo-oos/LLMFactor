from llmfactor.core.component.pipeline import PipelineStage, PipelineContext
from llmfactor.data.base import NewsProvider, PriceProvider
from llmfactor.core.component.registry import registry
from typing import List

@registry.register('data_fetch', 'v2')
class DataFetchStageV2(PipelineStage):
    """Fetches raw data from providers"""

    def __init__(self,
                 news_provider: NewsProvider,
                 price_provider: PriceProvider,
                 attribute: List[str],
                 price_k: int):
        super().__init__()
        self.news_provider = news_provider
        self.price_provider = price_provider
        self.price_k = price_k
        self.attribute = attribute

    def process(self, context: PipelineContext) -> PipelineContext:
        news = self.news_provider.get_news_by_date(
            context.ticker,
            context.target_date,
            attribute=self.attribute
        )
        if type(news) == List:
            context.raw_news = news
        elif type(news) == str:
            context.fake_news_text = news
        context.price_movements = self.price_provider.get_price_movements(
            context.ticker,
            context.target_date,
            self.price_k
        )
        return context
