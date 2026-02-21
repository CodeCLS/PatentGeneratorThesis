# tools/api/LLMApi_Repo.py
from __future__ import annotations
import os
from uuid import uuid4
from langfuse import Langfuse
from langfuse import observe
from tools.api.llm_models.anthropic_model import AnthropicModel  # adjust import if your paths differ
from tools.api.llm_models.gemini_model import GeminiModel  # adjust import if your paths differ
from tools.api.llm_models.gpt_oss_model import GPTOSSModel  # adjust import if your paths differ

from tools.api.base.base_llm_model import LLMModel

from tools.api.llm_models.deepseek_model import DeepSeekModel
from tools.api.llm_models.anthropic_model import AnthropicModel
from tools.api.llm_models.gemini_model import GeminiModel
from tools.api.llm_models.gpt_oss_model import GPTOSSModel

class LLmApi_Repo:
    def __init__(self, llm_client: LLMModel = DeepSeekModel(), test_mode: bool = False):
        self.test_mode = test_mode
        self.client = llm_client

    @staticmethod
    def get_available_models() -> List[str]:
        return ["deepseek", "anthropic", "gemini", "gpt-oss-20b"]

    def set_model(self, model_name: str):
        if model_name == "deepseek":
            self.client = DeepSeekModel()
        elif model_name == "anthropic":
            self.client = AnthropicModel()
        elif model_name == "gemini":
            self.client = GeminiModel()
        elif model_name == "gpt-oss-20b":
            self.client = GPTOSSModel()
        else:
            raise ValueError(f"Unknown model: {model_name}")

    @observe
    def chat(self, message: str, **kwargs) -> dict:
            if self.test_mode:
                return {"response": "Dummy response for testing.", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
            return self.client.send(message)

if __name__ == "__main__":
    repo = LLmApi_Repo()
    print(repo.chat("I am a dog and I am a cat."))
