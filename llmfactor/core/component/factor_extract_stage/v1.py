from llmfactor.core.component.pipeline import PipelineStage, PipelineContext
from llmfactor.util.llm_provider import LLMProvider
from llmfactor.util.format_text import extract_factors
from llmfactor.core.component.registry import registry

@registry.register('factor_extract', 'v1')
class FactorExtractStageV1(PipelineStage):
    """Extracts factors using LLM"""

    def __init__(self, llm_client: LLMProvider, factor_k: int):
        super().__init__()
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
