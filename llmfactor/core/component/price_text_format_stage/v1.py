from llmfactor.core.component.pipeline import PipelineStage, PipelineContext
from typing import Optional, List, Dict, Any, Tuple
from llmfactor.core.component.registry import registry


@registry.register('price_text_format', 'v1')
class PriceTextFormatStageV1(PipelineStage):
    """Formats raw data into text"""

    def __init__(self,
                 history_template: str = "On {date}, the stock price of {stock_target} {rise_fall}.\n",
                 target_template: str  = "On {date}, the stock price of {stock_target}"
                 ):
        super().__init__()
        self.history_template = history_template
        self.target_template = target_template

    def format_price_data(self, ticker: str, price_movements: Optional[List[Dict[str, Any]]]) -> Tuple[str, str]:
        if not price_movements:
            return "", ""

        price_history_text = ""
        for move in price_movements[:-1]:
            price_history_text += self.history_template.format(
                date=move['date'].strftime('%Y-%m-%d'),
                stock_target=ticker,
                rise_fall="rose" if move['rise'] else "fell"
            )

        price_target_text = self.target_template.format(
            date=price_movements[-1]['date'].strftime('%Y-%m-%d'),
            stock_target=ticker
        )

        return price_history_text, price_target_text


    def process(self, context: PipelineContext) -> PipelineContext:
        context.price_history_text, context.price_target_text = self.format_price_data(context.ticker, context.price_movements)
        return context
