
from sentence_splitter.SentenceSplitter import SentenceSplitter
class SentenceFormatterManager():
    def __init__(self):
        self.sentence_splitter = SentenceSplitter()
        pass
    def commit(self,text):
        return self.sentence_splitter.commit(text)