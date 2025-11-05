# kg/formatting/splitting/sentence_splitter.py
from typing import List
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
import ruptures as rpt

nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize


class SegmentSplitter:
    """Splits long text into context segments based on semantic/topic shifts."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def _embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """Convert each sentence into an embedding vector."""
        return self.model.encode(
            sentences, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        )

    def _detect_boundaries(self, embeddings: np.ndarray, pen: float = 6.0) -> List[int]:
        """Use change-point detection to find topic boundaries."""
        if len(embeddings) <= 2:
            return [len(embeddings)]

        algo = rpt.KernelCPD(kernel="rbf").fit(embeddings)
        breakpoints = algo.predict(pen=pen)
        if breakpoints[-1] != len(embeddings):
            breakpoints.append(len(embeddings))
        return breakpoints

    def run(self, text: str, pen: float = 6.0) -> List[str]:
        """
        Split text into segments when the context or topic changes.
        :param text: Input text to segment
        :param pen: Higher = fewer segments, Lower = more segments
        :return: List of text segments (strings)
        """
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        if not sentences:
            return []

        embeddings = self._embed_sentences(sentences)
        boundaries = self._detect_boundaries(embeddings, pen=pen)

        segments = []
        start = 0
        for end in boundaries:
            seg_text = " ".join(sentences[start:end]).strip()
            if seg_text:
                segments.append(seg_text)
            start = end

        return segments


# Example usage
if __name__ == "__main__":
    text = """
    The iPhone 15 introduces a new titanium frame and a USB-C port, which allows faster data transfer.
    Apple also claims improved battery life compared to the iPhone 14.
    Meanwhile, in the electric vehicle market, Tesla reported record deliveries this quarter.
    The company credited its new factory ramp for the increase.
    Separately, a rare comet will be visible in the northern hemisphere this weekend.
    Astronomers recommend dark-sky locations for the best view.
    """

    splitter = SegmentSplitter()
    segments = splitter.run(text)

    for i, seg in enumerate(segments, 1):
        print(f"\n--- Segment {i} ---\n{seg}")
