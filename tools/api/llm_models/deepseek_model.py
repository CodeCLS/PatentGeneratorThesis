# models/DeepSeekApiModel.py
from __future__ import annotations

from typing import Optional
from openai import OpenAI
import os
from tools.api.base.base_llm_model import LLMModel


class DeepSeekModel(LLMModel):
    def __init__(
        self,
        model: str = "deepseek-chat",
        system_prompt: str = ".",
        max_tokens: int = 500,
        base_url: str = "https://api.deepseek.com/v1",
    ):
        self.client = OpenAI(api_key=os.getenv("DEEPSEEK",""), base_url=base_url)
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"deepseek:{self.model}"

    def send(self, message: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": message},
            ],
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
