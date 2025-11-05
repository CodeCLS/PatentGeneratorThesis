# kg/formatting/splitting/sentence_splitter.py
from kg.formatting.splitting.sentence.sentence_splitter_agent import SentenceSplitterAgent


class SentenceSplitter:
    """Splits long patent-style text into shorter sentences without paraphrasing."""

    def __init__(self):
        self.agent = SentenceSplitterAgent(
            "Task: Split the following text into short, grammatically correct sentences. "
            "Do NOT paraphrase or replace any words. Preserve all original terminology and punctuation.\n"
            "Rules:\n"
            "- Keep all original wording and punctuation; do NOT rephrase or summarize.\n"
            "- Each element must be a short sentence (preferably ≤ 120 characters).\n"
            "- You may adjust punctuation to ensure grammatical correctness.\n"
            "- Return ONLY a JSON array of NEW sentences (strings), nothing else.\n"
            "- Example: [\"Sentence one.\", \"Sentence two.\"]\n\n"
        )

    def run(self, text: str):
        """Split the given patent text into multiple shorter sentences."""
        return self.agent.run(text)
