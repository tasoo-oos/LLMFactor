from llmfactor.core.component.pipeline import PipelineStage, PipelineContext
from llmfactor.util.llm_provider import LLMProvider
from llmfactor.core.component.registry import registry


@registry.register('price_analysis', 'v1')
class PriceAnalysisStageV1(PipelineStage):
    """Analyzes price movements using LLM"""

    def __init__(self, llm_client: LLMProvider):
        super().__init__()
        self.client = llm_client

    def process(self, context: PipelineContext) -> PipelineContext:
        factors = ""
        if type(context.extracted_factors) == list:
            factors = context.extracted_factors[-1]
        else:
            factors = context.extracted_factors

        messages = [
            {"role": "system",
             "content": f"Based on the following information, please judge the direction of the {context.ticker}'s stock price from rise/fall, fill in the blank and give reasons."},
        ]

        if factors:
            messages.append(
                {"role": "system",
                 "content": f"These are the main factors that may affect this stock's price recently:\n\n{factors}."},
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
