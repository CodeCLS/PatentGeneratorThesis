from typing import List
from tools.sentence.sentence import Sentence
from kg.formatting.splitting.sentence.sentence_splitter import SentenceSplitter
from kg.formatting.splitting.sentence.invention_sentence_extractor import InventionSentenceExtractor


class FormattingManager():
    def __init__(self):
        self.sentenceSplitter = SentenceSplitter()
        self.inventionExtractor = InventionSentenceExtractor()

    def split(self, sentences: List[Sentence]) -> List[Sentence]:
        """
        Split each Sentence object into multiple shorter sentences if necessary.
        
        Takes a list of Sentence objects, splits each one's text into shorter sentences,
        and returns a new list of Sentence objects. Each split sentence preserves metadata
        from the original sentence (source, tags, etc.) but gets a new index and ID.
        
        Args:
            sentences: List of Sentence objects to split
            
        Returns:
            List of Sentence objects, where each input sentence may be split into multiple
            shorter sentences
        """
        result: List[Sentence] = []
        global_index = 0
        
        for original_sentence in sentences:
            # Split the sentence text into shorter sentences
            split_sentences = self.sentenceSplitter.run(original_sentence.text)
            
            # Create new Sentence objects from the split results
            for split_sentence in split_sentences:
                # Create a new Sentence object preserving metadata from original
                new_sentence = Sentence(
                    text=split_sentence.text,
                    index=global_index,
                    source=original_sentence.source,
                    span=original_sentence.span,  # Keep original span
                    kg_node_id=original_sentence.kg_node_id,  # Keep original KG node ID if exists
                    embedding=original_sentence.embedding,  # Keep original embedding
                    tokens=original_sentence.tokens,  # Keep original tokens
                    entities=original_sentence.entities,  # Keep original entities
                    tags=original_sentence.tags.copy(),  # Copy tags
                    importance=original_sentence.importance,
                    info_quality=original_sentence.info_quality,
                    novelty=original_sentence.novelty,
                    lang=original_sentence.lang
                )
                result.append(new_sentence)
                global_index += 1
        
        return result
    
    def retrieveContent(self, patent_description: str):
        """
        Extract only sentences that refer to the invention from a patent description.
        Excludes introduction, examples, background, prior art, etc.
        
        Args:
            patent_description: Full patent description text
            
        Returns:
            List of Sentence objects containing only invention-related sentences
        """
        return self.inventionExtractor.run(patent_description)
 




if __name__ == "__main__":
    pass
