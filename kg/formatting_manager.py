from typing import List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tools.sentence.sentence import Sentence
from kg.sentence.sentence_splitter import SentenceSplitter
from kg.sentence.invention_sentence_extractor import InventionSentenceExtractor


class FormattingManager():
    def __init__(self, num_workers: int = 8, split_workers: int = 12):
        """
        Initialize FormattingManager.
        
        Args:
            num_workers: Number of workers for parallelizing retrieveContent() 
                        (invention sentence extraction via LLM calls). Default: 8
            split_workers: Number of workers for parallelizing split() 
                          (sentence splitting). Default: 12
        """
        self.sentenceSplitter = SentenceSplitter()
        self.inventionExtractor = InventionSentenceExtractor()
        self.num_workers = num_workers
        self.split_workers = split_workers

    def split(self, sentences: List[Sentence]) -> List[Sentence]:
        """
        Split each Sentence object into multiple shorter sentences if necessary.
        
        Takes a list of Sentence objects, splits each one's text into shorter sentences,
        and returns a new list of Sentence objects. Each split sentence preserves metadata
        from the original sentence (source, tags, etc.) but gets a new index and ID.
        Processing is done in parallel using ThreadPoolExecutor.
        
        Args:
            sentences: List of Sentence objects to split
            
        Returns:
            List of Sentence objects, where each input sentence may be split into multiple
            shorter sentences
        """
        if not sentences:
            return []
        
        # Helper function to split a single sentence
        def split_single_sentence(original_sentence: Sentence) -> List[Sentence]:
            # Split the sentence text into shorter sentences
            split_sentences = self.sentenceSplitter.run(original_sentence.text)
            
            # Create new Sentence objects from the split results
            result_sentences = []
            for split_sentence in split_sentences:
                # Create a new Sentence object preserving metadata from original
                new_sentence = Sentence(
                    text=split_sentence.text,
                    index=0,  # Will be set later with global index
                    source=original_sentence.source,
                    span=original_sentence.span,  # Keep original span
                    kg_node_id=original_sentence.kg_node_id,  # Keep original KG node ID if exists
                    embedding=original_sentence.embedding,  # Keep original embedding
                    tokens=original_sentence.tokens,  # Keep original tokens
                    entities=original_sentence.entities,  # Keep original entities
                    tags=original_sentence.tags.copy() if original_sentence.tags else {},  # Copy tags
                    importance=original_sentence.importance,
                    info_quality=original_sentence.info_quality,
                    novelty=original_sentence.novelty,
                    lang=original_sentence.lang
                )
                result_sentences.append(new_sentence)
            
            return result_sentences
        
        # Process sentences in parallel
        all_split_sentences = []
        with ThreadPoolExecutor(max_workers=self.split_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(split_single_sentence, sent): i 
                for i, sent in enumerate(sentences)
            }
            
            # Collect results as they complete, maintaining order
            results = [None] * len(sentences)
            for future in as_completed(future_to_index):
                sent_index = future_to_index[future]
                try:
                    split_results = future.result()
                    results[sent_index] = split_results
                except Exception as e:
                    print(f"Error splitting sentence {sent_index}: {e}")
                    results[sent_index] = []
        
        # Combine results in original order and assign global indices
        global_index = 0
        for split_results in results:
            if split_results:
                for sentence in split_results:
                    sentence.index = global_index
                    all_split_sentences.append(sentence)
                    global_index += 1
        
        return all_split_sentences
    
    def retrieveContent(self, patent_description: Union[str, List[str]], chunk_size: int = None):
        """
        Extract only sentences that refer to the invention from a patent description(s).
        Excludes introduction, examples, background, prior art, etc.
        
        If a single description is provided and chunk_size is specified, it will be split
        into chunks and processed in parallel. If a list of descriptions is provided,
        they will be processed in parallel.
        
        Args:
            patent_description: Full patent description text, or list of patent descriptions
            chunk_size: Optional. If provided and a single description is given, split it
                       into chunks of approximately this many CHARACTERS and process in parallel.
                       Example: chunk_size=5000 means chunks of ~5000 characters each.
                       If None and a single description is given, process it as-is.
            
        Returns:
            List of Sentence objects containing only invention-related sentences
        """
        # Handle list of patent descriptions - process in parallel
        if isinstance(patent_description, list):
            return self._retrieveContentParallel(patent_description)
        
        # Handle single description with optional chunking
        if chunk_size and len(patent_description) > chunk_size:
            chunks = self._split_into_chunks(patent_description, chunk_size)
            return self._retrieveContentParallel(chunks)
        
        # Single description, no chunking - process normally
        return self.inventionExtractor.run(patent_description)
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks of approximately chunk_size characters.
        Tries to split at paragraph boundaries when possible.
        """
        chunks = []
        if len(text) <= chunk_size:
            return [text]
        
        # Try to split at paragraph boundaries first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph would exceed chunk_size, save current chunk
            if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            
            # If a single paragraph is larger than chunk_size, split it by sentences
            if len(current_chunk) > chunk_size:
                sentences = current_chunk.split('. ')
                temp_chunk = ""
                for sent in sentences:
                    if temp_chunk and len(temp_chunk) + len(sent) + 2 > chunk_size:
                        chunks.append(temp_chunk.strip() + '.')
                        temp_chunk = sent
                    else:
                        if temp_chunk:
                            temp_chunk += '. ' + sent
                        else:
                            temp_chunk = sent
                current_chunk = temp_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _retrieveContentParallel(self, descriptions: List[str]) -> List[Sentence]:
        """
        Process multiple patent descriptions in parallel using ThreadPoolExecutor.
        
        Args:
            descriptions: List of patent description texts to process
            
        Returns:
            Combined list of Sentence objects from all descriptions
        """
        all_sentences = []
        global_index = 0
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_desc = {
                executor.submit(self.inventionExtractor.run, desc): i 
                for i, desc in enumerate(descriptions)
            }
            
            # Collect results as they complete
            results = [None] * len(descriptions)
            for future in as_completed(future_to_desc):
                desc_index = future_to_desc[future]
                try:
                    sentences = future.result()
                    results[desc_index] = sentences
                except Exception as e:
                    print(f"Error processing description {desc_index}: {e}")
                    results[desc_index] = []
            
            # Combine results in original order and reindex
            for sentences in results:
                if sentences:
                    for sentence in sentences:
                        # Create new sentence with updated global index
                        new_sentence = Sentence(
                            text=sentence.text,
                            index=global_index,
                            source=sentence.source,
                            span=sentence.span,
                            kg_node_id=sentence.kg_node_id,
                            embedding=sentence.embedding,
                            tokens=sentence.tokens,
                            entities=sentence.entities,
                            tags=sentence.tags.copy() if sentence.tags else {},
                            importance=sentence.importance,
                            info_quality=sentence.info_quality,
                            novelty=sentence.novelty,
                            lang=sentence.lang
                        )
                        all_sentences.append(new_sentence)
                        global_index += 1
        
        return all_sentences
 




if __name__ == "__main__":
    pass
