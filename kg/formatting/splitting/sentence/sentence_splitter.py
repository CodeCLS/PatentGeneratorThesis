# kg/formatting/splitting/sentence_splitter.py
from kg.formatting.splitting.sentence.sentence_splitter_agent import SentenceSplitterAgent


class SentenceSplitter:
    """
    Splits long patent-style text into shorter, standalone sentences,
    allowing controlled duplication of context so that each short sentence
    is meaningful on its own.
    """

    def __init__(self):
        self.agent = SentenceSplitterAgent(
            "You are a sentence splitting engine for patent text.\n"
            "Task: Transform the input into MULTIPLE short, COMPLETE sentences that are easy for NLP models to process.\n"
            "\n"
            "Rules:\n"
            "- Preserve ALL technical information from the original text. You may DUPLICATE words or phrases to keep context.\n"
            "- You MAY break up long 'comprising: X, Y, Z' lists into several sentences by repeating the subject.\n"
            "  Example:\n"
            "    Input: 'An electronic device, comprising: a first camera unit, a second camera unit, and a display unit.'\n"
            "    Allowed output:\n"
            "      [\n"
            "        'An electronic device comprising a first camera unit.',\n"
            "        'An electronic device comprising a second camera unit.',\n"
            "        'An electronic device comprising a display unit.'\n"
            "      ]\n"
            "- For long functional clauses (e.g. processing unit configured to: A; B; C;), you may create one sentence per function.\n"
            "  You may repeat 'the processing unit is configured to' so each sentence is standalone.\n"
            "- NEVER OMIT a technical element (e.g. a camera unit, display, processing step). Every element from the original\n"
            "  must appear in at least one of the output sentences.\n"
            "- Prefer more, shorter sentences over one very long sentence, as long as each is grammatically complete.\n"
            "- Avoid outputs that are just short fragments like 'For example.' or 'For the brick manufacturing industry.'\n"
            "- If needed to keep a sentence complete, REPEAT the subject or phrase (e.g. 'the device', 'the method').\n"
            "- Return ONLY a valid JSON array of strings (sentences), nothing else.\n"
            "- Example JSON: [\"The device comprises a first camera unit.\", \"The device comprises a second camera unit.\"]\n"
        )

    def run(self, text: str):
        """Split the given patent text into multiple short, standalone sentences."""
        return self.agent.run(text)
