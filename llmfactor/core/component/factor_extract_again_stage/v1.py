from llmfactor.core.component.pipeline import PipelineStage, PipelineContext
from llmfactor.util.llm_provider import LLMProvider
from llmfactor.util.format_text import extract_factors
from llmfactor.core.component.registry import registry
from typing import Literal

@registry.register('factor_extract_again', 'v1')
class FactorExtractAgainStageV1(PipelineStage):
    """Extracts factors using LLM"""

    def __init__(self,
                 llm_client: LLMProvider,
                 factor_k: int,
                 place_fake_text: Literal["first", "last"] = "last",
                 ) -> None:
        super().__init__()
        self.client = llm_client
        self.factor_k = factor_k
        self.place_fake_text = place_fake_text

    def process(self, context: PipelineContext) -> PipelineContext:
        if self.place_fake_text == "first":
            news = context.fake_news_text + "\n\n" + context.news_text + "\n\n"
        elif self.place_fake_text == "last":
            news = context.news_text + "\n\n" + context.fake_news_text + "\n\n"
        else:
            raise ValueError("Invalid value for place_fake_text. Must be 'first' or 'last'.")

        response = self.client.generate_completion(
            temperature=0,
            max_tokens=self.factor_k * 200,
            messages=[
                {"role": "system",
                 "content": f"Please extract the top {self.factor_k} factors that may affect the stock price of {context.ticker} from the following news."},
                {"role": "user", "content": "# news list\n\n" + news},
                {"role": "user",
                 "content": f"Now, tell me. What are the {self.factor_k} most influential market drivers for {context.ticker} based on recent news? Write your answer with following format:\n" + "".join(
                     [f"\n{i}." for i in range(1, self.factor_k + 1)])},
            ]
        )
        if context.extracted_factors:
            context.extracted_factors = [context.extracted_factors, extract_factors(response.content)]
        return context
