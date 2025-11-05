
from kg.formatting.splitting.segment.segment_splitter import SegmentSplitter

from kg.formatting.splitting.sentence.sentence_splitter import SentenceSplitter
from kg.formatting.simplifying.sentence_simplify import SentenceSimplifier

class FormattingManager():
    def __init__(self):
        self.sentenceSplitter = SentenceSplitter()
        self.sentenceSimplify = SentenceSimplifier()
        self.segmentSplitter = SegmentSplitter()

    def split(self, text : str):
        return self.sentenceSplitter.run(text)
    def simplify(self, text : str):
        return self.sentenceSimplify.run(text)
    def split_segments(self, text : str):
        return self.segmentSplitter.run(text)



if __name__ == "__main__":
    pass
