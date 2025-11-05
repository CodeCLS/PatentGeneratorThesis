import json
import sys, os
from langsmith.wrappers import wrap_anthropic  # traces anthropic calls
sys.path.append(os.path.abspath(".."))  # adds parent directory
from tools.api.llm_api_repo import LLmApi_Repo


PROMPT = """
You are a linguistic sentence splitter specialized in technical and patent text.

Your task:
Split the following input sentence into multiple shorter sentences
that preserve every word, term, and punctuation exactly as in the input.
Do NOT paraphrase, rephrase, summarize, or replace any words.

Rules:
- Keep all original wording, spacing (except leading/trailing whitespace), and punctuation.
- Do NOT split inside infinitive or participle phrases like "to adjust", "configured to", "being controlled".
- Do NOT add or remove words.
- If the sentence cannot be split naturally, return it unchanged as a single element in the JSON array.
- After the split, the sentence still needs to make grammatical sence

Return your answer **strictly** as a JSON array, for example:
[
  "First shorter sentence.",
  "Second shorter sentence.",
  "Third shorter sentence."
]

Input:
{input_text}
"""


class SentenceSplitter:
    def __init__(self):
        self.api_repo_llm = LLmApi_Repo()

    def commit(self, text: str):
        text = text.strip()
        if not text:
            return []

        prompt = PROMPT.replace("{input_text}", text)
        resp = self.api_repo_llm.chat(
            message=prompt,
            system="Respond only with a JSON array of strings. No explanations or markdown."
        )


        # 1) If provider already returned a Python list -> use it
        if isinstance(resp, list) and all(isinstance(x, str) for x in resp):
            return [s.strip() for s in resp if s.strip()]

        # 2) If dict: handle both new and legacy shapes
        if isinstance(resp, dict):
            if isinstance(resp.get("sentences"), list):
                return [s.strip() for s in resp["sentences"] if isinstance(s, str) and s.strip()]
            s1 = resp.get("sentence1", "")
            s2 = resp.get("sentence2", "")
            out = [s for s in (s1, s2) if isinstance(s, str) and s.strip()]
            if out:
                return out

        # 3) If string: try JSON, then safe Python literal (handles single quotes)
        if isinstance(resp, str):
            t = resp.strip()
            # Try JSON first
            if (t.startswith("[") and t.endswith("]")) or (t.startswith("{") and t.endswith("}")):
                try:
                    data = json.loads(t)
                    if isinstance(data, list) and all(isinstance(x, str) for x in data):
                        return [s.strip() for s in data if s.strip()]
                except json.JSONDecodeError:
                    pass
            # Try Python literal (e.g., "['...']" with single quotes)
            try:
                import ast
                lit = ast.literal_eval(t)
                if isinstance(lit, list) and all(isinstance(x, str) for x in lit):
                    return [s.strip() for s in lit if s.strip()]
            except Exception:
                pass

    

        print("[SentenceSplitter] Warning: falling back to original text. Raw:", type(resp), resp)
        return [text]



if __name__ == "__main__":
    splitter = SentenceSplitter()
    result = splitter.commit(
        "This is a test sentence that is quite long and perhaps contains some information about the model being used which is Anthropics model and I am currently at home."
    )
    for i, s in enumerate(result, start=1):
        print(f"{i:02d}. {s}")
