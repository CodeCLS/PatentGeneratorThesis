from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable, Deque, Tuple
from collections import deque
import json
import re
import anthropic
import pysbd


class SentenceSplitterAgent:
    """
    Anthropic-backed splitter that NEVER rewrites content.
    It only asks the model for sub-sentences and accepts boundaries.
    Optional tracing via `tracer` or `ls_client` (anything with .track/.log / callable).
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        max_len: int = 130,
        model: str = "claude-haiku-4-5",
        max_refine_iters: int = 3,           # hard cap per segment to avoid loops
        max_words: int = 30,                 # optional word-count constraint
        min_words: int = 6,                  # avoid tiny fragments (e.g., "and", "member")
    ):
        self.client = client
        self.max_len = max_len
        self.max_words = max_words
        self.min_words = min_words
        self.model = model
        self.max_refine_iters = max_refine_iters

    # ---------------- public API ----------------
    def shorten(self, text: str) -> List[str]:
        """
        Query the LLM for split sentences when needed.
        Returns a list of shorter, period-normalized sentences.
        Never paraphrases; only inserts boundaries.
        """
        text = text.strip()
        if not text:
            return []

        seg = pysbd.Segmenter(language="en", clean=True)
        parts = seg.segment(text)

        results: List[str] = []
        for part in parts:
            queue: Deque[Tuple[str, int]] = deque([(part.strip(), 0)])

            while queue:
                current, iters = queue.popleft()
                if not current:
                    continue

                # If already within constraints, accept
                if self._fits_constraints(current):
                    results.append(current)
                    continue

                # Cap refinement
                if iters >= self.max_refine_iters:
                    # Fallback: try pysbd, then last-resort hard cut
                    fallback_chunks = seg.segment(current)
                    for ch in fallback_chunks:
                        ch = ch.strip()
                        if not ch:
                            continue
                        if self._fits_constraints(ch):
                            results.append(ch)

                    continue

                # Ask the LLM for split suggestions (sentences)
                prompt = self._make_prompt(current)
                data = self._llm_json(prompt, model=self.model)

                candidates: List[str] = []
                if isinstance(data, dict) and isinstance(data.get("sentences"), list):
                    candidates = [s for s in data["sentences"] if isinstance(s, str) and s.strip()]

                # If nothing valid came back, fall back to pysbd split (non-rewriting)
                if not candidates:
                    candidates = seg.segment(current)

                any_pushed = False
                for cand in candidates:
                    cand = cand.strip()
                    if not cand:
                        continue
                    if self._fits_constraints(cand):
                        results.append(cand)
                    else:
                        queue.append((cand, iters + 1))
                        any_pushed = True

             
            

        return results

    # ---------------- internal helpers ----------------
    def _make_prompt(self, s: str) -> str:
        # Treat line breaks as spaces for the model (does not rewrite your final output)
        s_for_prompt = re.sub(r"\s*\n\s*", " ", s)
        return f"""You are a patent text splitting assistant.

Your task is to split the provided TEXT into multiple shorter sentences, each expressing a single complete idea or clause.
Do NOT paraphrase, summarize, or change any wording.
You may duplicate existing words if needed to make each sentence grammatically complete, but do not add or replace terms.
Preserve capitalization, terminology, and punctuation except for inserting periods at natural clause boundaries.

Rules for splitting:
- Split when multiple independent clauses are joined by words like "and", "but", "or", "which", "that", or similar.
- Split if a clause after "and" or "but" could form a standalone sentence (has its own subject and verb).
- Do not split inside enumerations, numeric expressions, units, or lists.
- If a sentence contains multiple consecutive actions or ideas, make each a separate sentence.
- Never change or remove words, only insert periods and adjust spacing.
- Each sentence MUST be under {self.max_len} characters AND at most {self.max_words} words, and at least {self.min_words} words.
- Avoid fragments: do not output single-word or function-word-only sentences. Prefer sentences with a subject and a verb.
- Treat line breaks as spaces; never split solely because of a line break.

Return only a JSON object of the form:
{{"sentences": ["<sentence1>", "<sentence2>", "<sentence3>", ...]}}

TEXT:
{s_for_prompt}
"""

    def _llm_json(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int = 1500,
    ) -> Dict[str, Any]:
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
        if getattr(resp, "content", None):
            block = resp.content[0]
            raw = getattr(block, "text", "") or (getattr(block, "input_text", "") or "")
        out = self._salvage_json(raw)
        return out

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

    def _fits_constraints(self, t: str) -> bool:
        t_stripped = t.strip()
        if not t_stripped:
            return True
        if len(t_stripped) > self.max_len:
            return False
        if self.max_words is not None and self._word_count(t_stripped) > self.max_words:
            return False
        if self.min_words is not None and self._word_count(t_stripped) < self.min_words:
            return False
        return True

    @staticmethod
    def _word_count(t: str) -> int:
        return len([w for w in t.strip().split() if w])


