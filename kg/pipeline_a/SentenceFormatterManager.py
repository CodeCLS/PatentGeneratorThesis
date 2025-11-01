
from kg.pipeline_a.sentence_splitter.SentenceParser import SentenceParser
class SentenceFormatterManager():
    def __init__(self):
        self.sentence_splitter = SentenceParser()
        pass
    def commit(self,text):
        return self.sentence_splitter.commit(text)