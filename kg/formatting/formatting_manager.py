
from kg.formatting.splitting.segment.segment_splitter import SegmentSplitter

from kg.formatting.splitting.sentence.sentence_splitter import SentenceSplitter
from kg.formatting.simplifying.sentence_simplify import SentenceSimplifier
from kg.formatting.rule_standardising.rule_standardiser import RuleStandardiser

class FormattingManager():
    def __init__(self):
        self.sentenceSplitter = SentenceSplitter()
        self.sentenceSimplify = SentenceSimplifier()
        self.segmentSplitter = SegmentSplitter()
        self.ruleStandardiser = RuleStandardiser()


    def split(self, text : str):
        return self.sentenceSplitter.run(text)
    def simplify(self, text : str): 
        return self.sentenceSimplify.run(text)
    def split_segments(self, text : str):
        return self.segmentSplitter.run(text)
    def rule_standardiser(self, text : str):
        return self.ruleStandardiser.run(text)




if __name__ == "__main__":
    pass
