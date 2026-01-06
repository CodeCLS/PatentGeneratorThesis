import os

from groq import Groq
from groq import (
    GroqError,
    APIError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
)


# models/AnthropicApiModel.py
import anthropic
import json
import re
from google import genai
from typing import Any, List
from tools.api.base.base_llm_model import LLMModel

# Non-greedy capture inside fences:
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)

# Non-greedy fallbacks:
FIRST_ARRAY_RE  = re.compile(r"\[(?:.|\n)*?\]")
FIRST_OBJECT_RE = re.compile(r"\{(?:.|\n)*?\}")

class GPTOSSModel(LLMModel):
    def __init__(self, model: str = "openai/gpt-oss-20b",system_prompt :str = "", max_tokens: int = 3000, api_key: str | None = None):
        
        self.client =Groq(
            api_key=os.environ.get("GROQ_API_KEY"
                                   ))
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"groq:{self.model}"


        
    def send(self, message: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": message},
                ],
            )

        except RateLimitError as e:
            # retry later
            raise RuntimeError("Groq rate limit hit") from e

        except AuthenticationError as e:
            # configuration error — do NOT retry
            raise RuntimeError("Groq authentication failed") from e

        except BadRequestError as e:
            # invalid model, too many tokens, bad params
            raise RuntimeError(f"Groq bad request: {e}") from e

        except GroqError as e:
            # catch-all for Groq SDK errors
            raise RuntimeError("Groq API error") from e

        # ---- success path ----
        try:
            return resp.choices[0].message.content
        except (IndexError, AttributeError):
            raise RuntimeError("Groq returned malformed response")
