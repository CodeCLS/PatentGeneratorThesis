import spacy
import benepar


class SentenceParser:
    """
    Splits long or complex sentences into smaller, clause-like sentences
    using benepar + spaCy.
    Keeps only top-level finite clauses (ignores embedded infinitives or 'while/that' clauses).
    """

    CLAUSE_LABELS = {"S", "SBAR", "SINV", "SQ"}

    def __init__(self, nlp=None):
        self.nlp = nlp or spacy.load("en_core_web_sm", disable=["ner", "parser"])

        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        if "benepar" not in self.nlp.pipe_names:
            try:
                self.nlp.add_pipe("benepar", config={"model": "benepar_en3_large"})
            except Exception:
                benepar.download("benepar_en3_large")
                self.nlp.add_pipe("benepar", config={"model": "benepar_en3_large"})

    # ---------- helpers ----------

    @staticmethod
    def _has_finite_verb(span):
        return any(
            t.pos_ in {"VERB", "AUX"} and "Fin" in t.morph.get("VerbForm", [])
            for t in span
        )

    # ---------- main ----------

    def commit(self, text: str):
        """Return only top-level, finite clauses."""
        text = text.strip()
        if not text:
            return []

        doc = self.nlp(text)
        results = []

        for sent in doc.sents:
            # 1. collect candidate clause spans
            cands = [
                cons for cons in sent._.constituents
                if any(lbl in self.CLAUSE_LABELS for lbl in cons._.labels)
            ]

            # 2. keep only top-level spans (not inside another)
            top_level = []
            for i, c in enumerate(sorted(cands, key=lambda x: (x.start, x.end))):
                if not any(
                    other.start <= c.start and c.end <= other.end and other != c
                    for other in cands
                ):
                    top_level.append(c)

            # 3. keep only finite, meaningful ones
            for span in top_level:
                if self._has_finite_verb(span) and len(span.text.split()) > 2:
                    results.append(span.text.strip())

            # 4. fallback: if nothing caught, take full sentence
            if not results and len(sent.text.split()) > 2:
                results.append(sent.text.strip())

        # 5. deduplicate / cleanup
        clean = []
        seen = set()
        for c in results:
            c = c.replace(" .", ".").replace(" ,", ",").strip()
            if c and c.lower() not in seen:
                seen.add(c.lower())
                clean.append(c)
        return clean


# ---------- test ----------
if __name__ == "__main__":
    splitter = SentenceSplitter()
    examples = [
        "A control module is configured to adjust motor speed and to send data to a remote device while the user operates the interface.",
        "The other end is so installed that it is in the liquid when the liquid is filled in a water tank for appreciation.",
    ]

    for text in examples:
        print("\nInput:", text)
        for i, s in enumerate(splitter.commit(text), start=1):
            print(f"  {i:02d}. {s}")
