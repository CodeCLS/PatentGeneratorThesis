from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json, ast
from tools.api.llm_models.anthropic_model import AnthropicModel
class RecursivePromptingAgent(ABC):

    @abstractmethod
    def __init__(self, prompt : str):
        super().__init__()
    
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def initial_state(self, seed: Any) -> Dict[str, Any]:
        """Return a mutable state dict for the loop."""
        ...

    @abstractmethod
    def build_prompt(self, state: Dict[str, Any]) -> str:
        """Produce the next prompt from current state."""
        ...

    @abstractmethod
    def handle_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        """Parse LLM response and update state."""
        ...

    
    @abstractmethod
    def run(seed: str) -> Any:
        """Run the Agent"""
        ...

    @abstractmethod
    def should_stop(self, state: Dict[str, Any], iteration: int) -> bool:
        """Return True to stop the loop."""
        ...

    @abstractmethod
    def result(self, state: Dict[str, Any]) -> Any:
        """Return the final artifact from state."""
        ...

    @property
    def llm(self):
        if self._llm is None:
            return AnthropicModel()
        return self._llm
    
    @llm.setter
    def llm(self, llm_client):
        self._llm = llm_client

