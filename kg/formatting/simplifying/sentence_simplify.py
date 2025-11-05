# kg/formatting/simplifying/sentence_simplifier.py
from kg.formatting.simplifying.sentence_simplify_agent import SentenceSimplifyAgent


class SentenceSimplifier:
    def __init__(self):
        self.agent = SentenceSimplifyAgent(
           "You rewrite the following SINGLE sentence into a simpler WORDED sentence.\n"
            "Rules (NO information loss):\n"
            "- Keep ALL facts, conditions, constraints, references, and entities.\n"
            "- You may simplify wording and grammar and simplify sentences if the content and information stays the same and there is no data loss.\n"
            "- Do NOT remove or change numbers, measurement units, ranges, or limits.\n"
            "- Do NOT rename technical terms; keep domain terms verbatim.\n"
            "- Preserve quoted phrases, bracketed tags, and parenthetical content.\n"
            "- Output MUST be exactly ONE sentence.\n"
        )

    def run(self, sentence: str):
        print("Run sentence simplifier")
        result = self.agent.run(sentence)
        return result
