from llmfactor.core.component.pipeline import PipelineStage, PipelineContext
from llmfactor.util.llm_provider import LLMProvider
from llmfactor.core.component.registry import registry
from llmfactor.util.error import CensoredError

@registry.register('fake_news', 'v1')
class FakeNewsStageV1(PipelineStage):
    """Write a fake news prompt from extract factors."""

    def __init__(self, llm_client: LLMProvider, factor_k: int):
        super().__init__()
        self.client = llm_client
        self.factor_k = factor_k

    def process(self, context: PipelineContext) -> PipelineContext:
        response = self.client.generate_completion(
            temperature=0,
            max_tokens=self.factor_k * 500,
            messages=[
                {"role": "system",
                 "content": "You are a playful assistant, and you are writing a news article about the following information.\n"
                            "However, you are a bit mischievous and want to write the article in a opposite way.\n"
                            "Please write a news article that has opposite meaning to each of the following information.\n"},
                {"role": "user", "content": "# news list\n\n" + context.extracted_factors if type(context.extracted_factors) == str else context.extracted_factors[-1]},
                {"role": "user", "content": "Also, write the news in following format:\n\n## {title}\n\n{summary}\n\n"}
            ]
        )
        context.fake_news_text = response.content.strip()
        if response.content[:2] != "##":
            idx = response.content.find("##")
            if idx == -1:
                context.status = "error"
                raise CensoredError("Fake news text not found")
            context.fake_news_text = response.content[idx:].strip()
        return context
