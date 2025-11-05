# kg/api/base/LLMModel.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class LLMModel(ABC):

    """Abstract base for all LLM wrappers (Anthropic, Gemini, etc.)"""
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def send(self, prompt: str) -> str:
        """Return the model's text response for a single prompt."""