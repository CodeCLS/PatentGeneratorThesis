
from sentence_splitter import SentenceSplitter
from sentence_splitter import SentenceParser

class SentenceFormatterManager():
    def __init__(self):
        self.sentence_splitter = SentenceSplitter()
        self.sentence_parser = SentenceParser()

        pass
    def split(self,text):
        return self.sentence_splitter.commit(text)
    def simplify(self,text):
        return self.sentence_parser.commit(text)
    
if __name__ == "__main__":
    SentenceFormatterManager().split("")