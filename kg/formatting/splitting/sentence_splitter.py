from kg.formatting.splitting.sentence_splitter_agent import SentenceSplitterAgent

class SentenceSplitter:
    def __init__(self):
        self.agent = SentenceSplitterAgent("Task: Split the following text into short sentences (<= 120 chars). "
            "Do NOT paraphrase. Keep all original words and punctuation.")
    def run(self):
        return self.agent.run()