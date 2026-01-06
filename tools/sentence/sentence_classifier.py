"""
Sentence classifier for filtering informative sentences.
Supports batch processing and GPU acceleration.
"""
from typing import List, Optional, Tuple, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tools.sentence.sentence import Sentence


class SentenceClassifier:
    """
    Classifies sentences as informative or not using a pre-trained model.
    Supports batch processing for GPU efficiency and parallel processing for CPU.
    """
    
    def __init__(
        self, 
        model_path: str = "training/info/done/hf/sentence_classifier_model",
        batch_size: int = 32,
        use_gpu: bool = True,
        max_length: int = 256
    ):
        """
        Initialize the sentence classifier.
        
        Args:
            model_path: Path to the pre-trained model directory
            batch_size: Batch size for processing (larger = faster on GPU, more memory)
            use_gpu: Whether to use GPU if available
            max_length: Maximum token length for input sentences
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Setup device (GPU if available and requested)
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Get label mapping
        self.id2label = self.model.config.id2label
        
        print(f"SentenceClassifier initialized on device: {self.device}")
    
    def classify(self, text: str) -> Tuple[str, List[float]]:
        """
        Classify a single sentence.
        
        Args:
            text: Sentence text to classify
            
        Returns:
            Tuple of (label_name, probabilities_list)
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = outputs.logits.softmax(dim=-1)[0]
        pred_id = int(torch.argmax(probs))
        label_name = self.id2label[pred_id]
        
        return label_name, probs.cpu().tolist()
    
    def classify_batch(self, texts: List[str]) -> List[Tuple[str, List[float]]]:
        """
        Classify multiple sentences in a batch (much faster on GPU).
        
        Args:
            texts: List of sentence texts to classify
            
        Returns:
            List of tuples (label_name, probabilities_list) for each sentence
        """
        if not texts:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Classify batch
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get predictions and probabilities
            probs = outputs.logits.softmax(dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            
            # Convert to results
            for j, pred_id in enumerate(pred_ids):
                label_name = self.id2label[int(pred_id)]
                prob_list = probs[j].cpu().tolist()
                results.append((label_name, prob_list))
        
        return results
    
    def filter_informative(
        self, 
        sentences: List[Sentence], 
        keep_labels: Optional[List[str]] = None
    ) -> List[Sentence]:
        """
        Filter sentences to keep only informative ones.
        
        Args:
            sentences: List of Sentence objects to filter
            keep_labels: List of labels to keep (default: ["INFORMATIVE"])
            
        Returns:
            Filtered list of Sentence objects
        """
        if not sentences:
            return []
        
        if keep_labels is None:
            keep_labels = ["INFORMATIVE"]
        
        # Extract texts for batch processing
        texts = [sentence.text for sentence in sentences]
        
        # Classify all sentences in batches
        classifications = self.classify_batch(texts)
        
        # Filter sentences based on labels
        filtered = []
        for sentence, (label_name, probs) in zip(sentences, classifications):
            if label_name in keep_labels:
                filtered.append(sentence)
        
        return filtered
    
    def classify_sentences(
        self, 
        sentences: List[Sentence]
    ) -> List[Tuple[Sentence, str, List[float]]]:
        """
        Classify a list of Sentence objects and return results with sentences.
        
        Args:
            sentences: List of Sentence objects to classify
            
        Returns:
            List of tuples (sentence, label_name, probabilities_list)
        """
        if not sentences:
            return []
        
        texts = [sentence.text for sentence in sentences]
        classifications = self.classify_batch(texts)
        
        return [
            (sentence, label_name, probs)
            for sentence, (label_name, probs) in zip(sentences, classifications)
        ]

