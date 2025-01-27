from llmfactor.core.component.pipeline import PipelineStage, PipelineContext
from llmfactor.core.component.registry import registry


@registry.register('resolve_prediction', 'v1')
class ResolvePredictionStageV1(PipelineStage):
    """Extracts prediction from analysis result"""

    def __init__(self):
        super().__init__()

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
