from abc import ABC, abstractmethod
from openai import OpenAI
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str


class LLMProvider(ABC):
    @abstractmethod
    def generate_completion(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        pass


class OpenAIProvider(LLMProvider):
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model

    def generate_completion(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return LLMResponse(content=response.choices[0].message.content)


class LLMProviderFactory(ABC):
    """Factory for creating llm endpoint providers based on configuration"""

    @staticmethod
    def create_llm_provider(provider_type: str, **kwargs) -> LLMProvider:
        if provider_type == "openai":
            return OpenAIProvider(**kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")