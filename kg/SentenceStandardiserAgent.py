from __future__ import annotations
from typing import List, Dict, Any, Optional
import json, re, anthropic


class SentenceStandardiserAgent:
    """
    Generic LLM-based Reference Standardiser.

    - Works on any type of reference ("one end", "terminal", "side", "opening", etc.).
    - Does not assume specific keys like EndA/EndB.
    - The LLM is fully responsible for naming, linking, and structuring references.
    - The Python class only:
        1. Calls the LLM sequentially per sentence.
        2. Maintains and merges a generic context dict.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-haiku-4-5",
        max_refine_iters: int = 2,
        known_entities: Optional[Dict[str, str]] = None,
        entity_aliases: Optional[Dict[str, str]] = None,
    ):
        self.client = client
        self.model = model
        self.max_refine_iters = max_refine_iters
        self.known_entities = known_entities or {}
        self.entity_aliases = entity_aliases or {}

    # ---------------- main public API ----------------
    def standardise(self, sentences: List[str]) -> List[str]:
        """
        Processes a list of sentences sequentially with rolling LLM context.
        Returns the standardised sentences.
        """
        context = {}
        results = []

        for sent in sentences:
            prompt = self._make_prompt(sent, context)
            std_sentence, context = self._run_llm(prompt, context)
            results.append(std_sentence)
        return results

    # ---------------- internal helpers ----------------
    def _run_llm(self, prompt: str, context: Dict[str, Any]) -> (str, Dict[str, Any]):
        """
        Run a few LLM retries if needed. Merge whatever context the model returns.
        """
        for _ in range(self.max_refine_iters):
            data = self._llm_json(prompt)
            if not isinstance(data, dict):
                continue
            if "sentence" in data and isinstance(data["sentence"], str):
                updated_context = self._merge_context(context, data.get("context_update", {}))
                return data["sentence"].strip(), updated_context
        # fallback: return unchanged if LLM failed
        return prompt, context

    def _make_prompt(self, sentence: str, context: Dict[str, Any]) -> str:
        ctx_json = json.dumps(context, ensure_ascii=False)
        hints = json.dumps({
            "known_entities": self.known_entities,
            "entity_aliases": self.entity_aliases
        }, ensure_ascii=False)
        s = re.sub(r"\s*\n\s*", " ", sentence)

        return f"""You are a patent reference standardiser.

Your job is to normalise references across sentences (e.g., "[Object]","[Object.Start]","[Object.End]") including them in capital and brackets [Reference].
You receive one sentence at a time plus a shared CONTEXT from previous sentences.

Rules:
- Do not paraphrase or reword anything.
- Only modify reference phrases to ensure consistent, canonical naming.
- Example: if one sentence says "one end of the connector", you might name it "[Connector.EndA]",
  and if a later sentence says "the other end", rename it to "[Connector.EndB]".
- However, the reference type and naming scheme (EndA, Terminal1, SideB, etc.) are entirely up to you.
- The CONTEXT tells you what references already exist; update it as needed.
- Keep punctuation and structure intact.
- Sometimes the wording will describe something in high detail such as "A smelly watery yellowish pipe", you can ignore the adjectives and just call it in capital [PIPE]
- Return only valid JSON like this:
{{
  "sentence": "<rewritten sentence>",
  "context_update": {{
     "<entity_or_reference_name>": {{
         "type": "<category or None>",
         "links": ["<related references or None>"],
         "resolved_names": ["<canonical forms>"]
     }}
  }}
}}

CONTEXT:
{ctx_json}

ENTITY_HINTS:
{hints}

SENTENCE:
{s}
"""

    def _llm_json(self, prompt: str) -> Dict[str, Any]:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system="Respond only with a single JSON object. No explanations, no markdown.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text if getattr(resp, "content", None) else ""
        try:
            return json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            return json.loads(m.group(0)) if m else {}

    @staticmethod
    def _merge_context(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generic recursive merge that doesn’t assume any schema.
        The LLM decides the structure; we just union keys.
        """
        if not update:
            return base
        merged = dict(base)
        for k, v in update.items():
            if k not in merged:
                merged[k] = v
            elif isinstance(v, dict) and isinstance(merged[k], dict):
                merged[k] = SentenceStandardiserAgent._merge_context(merged[k], v)
            else:
                merged[k] = v
        return merged
