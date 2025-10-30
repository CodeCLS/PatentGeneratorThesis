from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable
import json
import re
import anthropic
import nltk
from nltk.tokenize import sent_tokenize

class SentenceSplitterLLM:
    """
    Anthropic-backed splitter that NEVER rewrites content.
    It only asks the model for split indices and slices the original text.
    Optional tracing via `ls_client` (anything with .track/.log / callable).
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        max_len: int = 220,
        model: str = "claude-haiku-4-5",
        tracer: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        ls_client: Optional[object] = None,  # e.g., your LSClient()
    ):
        self.client = client
        self.max_len = max_len
        self.model = model
        self.tracer = tracer
        self.ls_client = ls_client

    # ---------------- public API ----------------
    def shorten(self, text: str) -> List[str]:
        """
        If `text` is too long, query the LLM for split sentences or indices.
        Returns a list of shorter, period-normalized sentences.
        """
        text = text.strip()
        if not text:
            return []
        parts = sent_tokenize(text)
        result = []
        for part in parts:
            prompt = self._make_prompt(part)
            data = self._llm_json(prompt, model=self.model)

            # Extract sentences from JSON (ignore missing or invalid fields)
            sentences = data.get("sentences", [])
            if isinstance(sentences, list):
                # Add each sentence from this part to the result list
                for s in sentences:
                    if isinstance(s, str) and s.strip():
                        result.append(s.strip())

        return result


    # ---------------- internal helpers ----------------
    def _make_prompt(self, s: str) -> str:
        return f"""You are a patent text splitting assistant.

Your task is to split the provided TEXT into multiple shorter sentences, each expressing a single complete idea or clause.
Do NOT paraphrase, summarize, or change any wording.
You may duplicate existing words if needed to make each sentence grammatically complete, but do not add or replace terms.
Preserve capitalization, terminology, and punctuation except for inserting periods at natural clause boundaries.

Rules for splitting:
- Split when multiple independent clauses are joined by words like "and", "but", "or", "which", "that", or similar.
- Split if a clause after "and" or "but" could form a standalone sentence (contains its own subject and verb).
- Do not split inside enumerations, numeric expressions, units, or lists.
- If a sentence contains multiple consecutive actions or ideas, make each a separate sentence.
- Never change or remove words, only insert periods and adjust spacing.
- Sentence Length MUST be under 80 Characters and if it is not, it MUST be split into another sentence. DON'T REMOVE INFORMATION AND MAKE SURE IT IS UNDER 80 Characters or 18 Words

Return only a JSON object of the form:
{{"sentences": ["<sentence1>", "<sentence2>", "<sentence3>", ...]}}

TEXT:
    {s}
    """


    def _llm_json(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int = 1500,
    ) -> Dict[str, Any]:
        """
        Calls Anthropic and expects strict JSON. Enforced via system message;
        if the model emits noise, try to salvage the JSON object.
        """
        resp = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=(
                "You are a disciplined JSON generator. "
                "Always respond with a single JSON object and nothing else. "
                "No prose, no backticks, no explanations."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        raw = ""
        # Anthropic SDK returns content blocks; take first text block if present
        if getattr(resp, "content", None):
            block = resp.content[0]
            raw = getattr(block, "text", "") or (getattr(block, "input_text", "") or "")
        return self._salvage_json(raw)

    @staticmethod
    def _salvage_json(raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return {}
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}

    @staticmethod
    def _sanitize_indices(indices: List[int], s: str) -> List[int]:
        """
        Ensure indices are strictly increasing, in-range, and not splitting mid-token.
        """
        clean = sorted(set(i for i in indices if 0 < i < len(s) - 1))
        safe: List[int] = []
        for i in clean:
            left, right = s[i - 1], s[i]
            # avoid splitting numbers/IDs or plain alpha tokens
            if (left.isdigit() and right.isdigit()) or (left.isalpha() and right.isalpha()):
                continue
            safe.append(i)
        return safe

    @staticmethod
    def _ensure_period(t: str) -> str:
        t = t.strip()
        return t if (not t or t.endswith(".")) else t + "."

    def _trace(self, name: str, data: Dict[str, Any]) -> None:
        """
        Best-effort tracing:
        - If `tracer` is a callable, call tracer(name, data).
        - Else if `ls_client` has `track` / `log` / `track_event`, call it.
        - Otherwise no-op.
        """
        if callable(self.tracer):
            try:
                self.tracer(name, data)
                return
            except Exception:
                pass
        if self.ls_client is not None:
            for method in ("track", "log", "track_event"):
                fn = getattr(self.ls_client, method, None)
                if callable(fn):
                    try:
                        fn(name, data)
                        return
                    except Exception:
                        return
        # silent no-op
