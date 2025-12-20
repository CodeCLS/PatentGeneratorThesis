

from kg.formatting.splitting.sentence.sentence_splitter import SentenceSplitter


class FormattingManager():
    def __init__(self):
        self.sentenceSplitter = SentenceSplitter()

    def split(self, text : str):
        return self.sentenceSplitter.run(text)
 




if __name__ == "__main__":
    pass
