from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable
import json
import re
import anthropic


class BoilerplateRemoverAgent:
    """
    Anthropic-backed boilerplate remover.
    - Processes sentences one-by-one.
    - Does NOT split/merge sentences.
    - Removes only boilerplate/filler while preserving meaning and order.
    - If LLM output is empty or malformed, falls back to a conservative regex scrub.
    - Optional tracing via `tracer` (callable) or any `ls_client` with .track/.log/.track_event.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        max_len: int = 130,              # not enforced, kept for compatibility if you want to nudge prompt later
        model: str = "claude-haiku-4-5",
        max_words: int = 30,             # not enforced, kept for compatibility
        min_words: int = 1,              # allow 1+ after removal
        max_refine_iters: int = 2,       # retry a couple of times if LLM output is bad
        tracer: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        ls_client: Optional[object] = None,
    ):
        self.client = client
        self.model = model
        self.max_len = max_len
        self.max_words = max_words
        self.min_words = min_words
        self.max_refine_iters = max_refine_iters
        self.tracer = tracer
        self.ls_client = ls_client

        # conservative boilerplate patterns (fallback only)
        self._boilerplate_patterns = [
            r"\bthe\s+present\s+invention\b",
            r"\baccording\s+to\s+(the\s+)?present\s+invention\b",
            r"\bin\s+one\s+embodiment\b",
            r"\bin\s+this\s+embodiment\b",
            r"\bin\s+an\s+embodiment\b",
            r"\bby\s+way\s+of\s+example\b",
            r"\bfor\s+example\b",
            r"\bfor\s+instance\b",
            r"\bas\s+such\b",
            r"\bin\s+particular\b",
            r"\btherefore\b",
            r"\bpreferably\b",
            r"\bmay\s+be\b",                      # careful: removes filler 'may be' disclaimers; keep conservative
            r"\bcan\s+be\b",
            r"\bshould\b",
            r"\bwherein\b",                       # common legal connective, often redundant
            r"\bwhereby\b",
            r"\bsaid\b(?=\s+\w)",                 # 'said X' -> 'X'
        ]

    # ---------------- public API ----------------
    def removeBoilerplate(self, sentences: List[str] | str) -> List[str]:
        """
        Remove boilerplate/filler from each sentence, preserving sentence boundaries and meaning.
        Accepts a list of sentences (recommended) or a single string (treated as one sentence).
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        results: List[str] = []
        for i, sent in enumerate(sentences):
            cur = sent.strip()
            if not cur:
                results.append(cur)
                continue

            cleaned = self._llm_clean_sentence(cur)
            if not cleaned:
                # Fallback to conservative regex scrub if LLM gave nothing useful
                cleaned = self._fallback_scrub(cur)

            # Final tidy: normalize spaces without altering words
            cleaned = self._normalize_spaces(cleaned)

            # Never allow empty string replacement; if empty, keep original to avoid information loss
            if not cleaned.strip():
                cleaned = cur

            results.append(cleaned)

            self._trace("bp_done_sentence", {
                "index": i,
                "original": cur,
                "cleaned": cleaned
            })

        return results

    # ---------------- LLM path ----------------
    def _llm_clean_sentence(self, sentence: str) -> str:
        """
        Calls the LLM with the fixed prompt (unchanged), enforces single-sentence return,
        retries a bit if malformed, otherwise returns '' to trigger fallback.
        """
        prompt = self._make_prompt(sentence)
        out_text = ""

        for attempt in range(self.max_refine_iters + 1):
            data = self._llm_json(prompt, model=self.model)
            sents = self._extract_sentences_array(data)

            # Expect exactly one sentence back (we send one in)
            if len(sents) == 1 and sents[0].strip():
                out_text = sents[0].strip()
                break

            self._trace("bp_retry_malformed", {
                "attempt": attempt,
                "returned_count": len(sents),
                "data_keys": list(data.keys()) if isinstance(data, dict) else "non-dict"
            })

        return out_text

    def _make_prompt(self, s: str) -> str:
        # Treat line breaks as spaces when asking the model (does not rewrite your output)
        s_for_prompt = re.sub(r"\s*\n\s*", " ", s)
        return f"""You are a patent text cleanup assistant.

Your task is to identify and remove boilerplate, filler, or redundant words from the provided TEXT.
These include words or phrases that do not contribute technical or structural meaning, such as generic connectors, formalities, or repetitive expressions.

Do NOT paraphrase, summarize, or replace any wording.
Do NOT add or invent new terms.
Do NOT change the order of words or alter the sentence meaning.
If removing a word causes a grammatical issue, you may minimally fix spacing or punctuation — but never rewrite or restate.
Each resulting sentence must remain grammatically natural and preserve all original information.

Guidelines:
- Remove only words that add no informational value (e.g., “in particular”, “as such”, “in this embodiment”, “preferably”, “for example”, “therefore”, “said”, “the present invention”, etc.).
- Keep all technical nouns, verbs, and structural relations intact.
- Do not merge, split, or rephrase sentences.
- Do not change capitalization or terminology.
- If uncertain whether a word is boilerplate or meaningful, KEEP it.

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
        max_tokens: int = 800,
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
        self._trace("llm_raw", {"raw": raw, "parsed_ok": bool(out)})
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

    @staticmethod
    def _extract_sentences_array(data: Any) -> List[str]:
        if isinstance(data, dict) and isinstance(data.get("sentences"), list):
            return [s for s in data["sentences"] if isinstance(s, str)]
        return []

    # ---------------- conservative fallback ----------------
    def _fallback_scrub(self, s: str) -> str:
        t = s
        # remove 'said X' -> 'X'
        t = re.sub(r"\bsaid\s+(?=\w)", "", t, flags=re.IGNORECASE)
        # remove listed boilerplate phrases
        for pat in self._boilerplate_patterns:
            t = re.sub(pat, "", t, flags=re.IGNORECASE)
        # collapse leftover double spaces and fix spaces before punctuation
        t = self._normalize_spaces(t)
        t = re.sub(r"\s+([,.;:])", r"\1", t)
        return t.strip()

    @staticmethod
    def _normalize_spaces(t: str) -> str:
        return re.sub(r"\s+", " ", t).strip()

    # ---------------- tracing ----------------
    def _trace(self, name: str, data: Dict[str, Any]) -> None:
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
