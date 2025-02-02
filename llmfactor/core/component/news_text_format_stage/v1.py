from llmfactor.core.component.pipeline import PipelineStage, PipelineContext
from typing import Optional, List, Dict, Any
from llmfactor.core.component.registry import registry

@registry.register('news_text_format', 'v1')
class NewsTextFormatStageV1(PipelineStage):
    """Formats raw data into text"""

    def __init__(self, format_template: str = "## {title}\n{summary}\n\n"):
        super().__init__()
        self.format_template = format_template

    def format_news_data(self, raw_news: Optional[List[Dict[str, Any]]]) -> str:
        news_text = ""
        for news in raw_news:
            news_text += self.format_template.format(
                title=news['title'],
                summary=news['summary']
            )
        return news_text.strip()


    def process(self, context: PipelineContext) -> PipelineContext:
        context.news_text = self.format_news_data(context.raw_news)
        return context
