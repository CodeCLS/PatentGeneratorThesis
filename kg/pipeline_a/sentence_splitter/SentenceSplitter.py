import json
import sys, os
from langsmith.wrappers import wrap_anthropic  # traces openai calls
sys.path.append(os.path.abspath(".."))  # adds parent directory
from tools.api.LLMApi_Repo import LLmApi_Repo

PROMPT = """
You are a linguistic sentence splitter specialized in technical and patent text.

Your task:
Split the following input sentence into TWO shorter sentences
that preserve every word, term, and punctuation exactly as in the input.
Do NOT paraphrase, rephrase, summarize, or replace any words.

Rules:
- Keep all original wording, spacing (except leading/trailing whitespace), and punctuation.
- Only insert a split where the grammar clearly allows a new independent clause 
  (typically where a new finite verb begins, e.g. "is", "was", "can", "will", "has").
- Do NOT split inside infinitive or participle phrases like "to adjust", "configured to", "being controlled".
- Do NOT add or remove words.
- If the sentence cannot be split naturally into two finite clauses, return it unchanged in "sentence1" and leave "sentence2" empty.

Return your answer as a JSON object with the following fields:
{
  "sentence1": "<first sentence>",
  "sentence2": "<second sentence or empty string if not applicable>"
}

Input:
{input_text}
"""

class SentenceSplitter:
    def __init__(self):
        self.api_repo_llm = LLmApi_Repo()

    def commit(self, text: str):
        prompt = PROMPT.replace("{input_text}", text.strip())
        obj = self.api_repo_llm.chat(message=prompt)  # -> dict: {"sentence1": "...", "sentence2": "..."}

        try:
            s1 = obj.get("sentence1", "").strip()
            s2 = obj.get("sentence2", "").strip()
            print("s1: " + str(s1) + " " + "s2: " + str(s2))
            out = [s for s in (s1, s2) if s]
            return out if out else [text.strip()]
        except Exception as e:
            print(f"[SentenceSplitter] Warning: failed to use LLM output ({e})")
            print("Raw LLM output:", obj)
            return [text.strip()]

if __name__ == "__main__":
    splitter = SentenceSplitter()
    result = splitter.commit(
        "This is a test sentence that is quite long and perhaps contains some information about the model being used which is Anthropics model and I am currently at home."
    )
    for i, s in enumerate(result, start=1):
        print(f"{i:02d}. {s}")
