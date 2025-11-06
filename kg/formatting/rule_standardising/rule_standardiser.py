# kg/formatting/simplifying/sentence_simplifier.py
from kg.formatting.rule_standardising.rule_standardiser_agent import RuleStandardiserAgent


class RuleStandardiser:
    def __init__(self):
        self.agent = RuleStandardiserAgent(
           """You rewrite the following SINGLE sentence to enforce the rules below.

Return format:
- Return ONLY a JSON array with EXACTLY one string: ["..."].

Rules (NO information loss):
- Keep ALL facts, conditions, constraints, references, entities, and list order.
- Convert passive to active. Do NOT invent new actors. If the actor is not stated, rewrite so the system/component already mentioned becomes the subject (e.g., "the controller regulates …").
- Preserve quoted phrases, bracketed tags (e.g., [DEVICE]), and parenthetical content exactly.
- Do NOT change or drop numbers, measurement units, ranges, or limits.
- Do NOT rename or synonymize technical/domain terms; keep them verbatim.
- If the sentence contains unverified claims, hedges, or speculation, prefix with [ASSUMPTION] (keep the rest unchanged except passive→active).
  Cues: "may", "might", "could", "intended to", "believed to", "assumed to", "expected to", "approximately", "about".
- If the sentence is generic or adds no useful information, prefix with [GENERIC].
  Cues: marketing/boilerplate like "state-of-the-art", "high quality", "in some embodiments" without specifics, or tautologies.
- Output MUST be exactly ONE sentence (no splitting, no merging). Minimal syntactic edits only to achieve active voice and tagging.

Validation checklist (the model must self-comply):
- One JSON array, length 1.
- No new entities/actors introduced.
- All numbers/units unchanged.
- Quoted/bracketed/parenthetical content untouched.

Examples:
Input: "The valve is configured to maintain 2–3 bar."
Output: ["The valve maintains 2–3 bar."]

Input: "The device is believed to be configured to regulate flow."
Output: ["[ASSUMPTION] The device regulates flow."]

Input: "The product is designed for superior experience."
Output: ["[GENERIC] The product provides a superior experience."]

Input: "The data were processed by the module."
Output: ["The module processes the data."]
"""
        )

    def run(self, sentence: str):
        print("Run sentence simplifier")
        result = self.agent.run(sentence)
        return result
