from transformers import AutoTokenizer, AutoModel
import torch


class TextEncoder:
    def __init__(self):
        MODEL_NAME = "anferico/bert-for-patents"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

        print("Encoder using device:", self.device)

    def mean_pooling(self,model_output, attention_mask):
        """
        Standard sentence embedding pooling:
        Masked mean pooling over last_hidden_state.
        """
        token_embeddings = model_output.last_hidden_state  # [B, L, H]
        mask = attention_mask.unsqueeze(-1).float()        # [B, L, 1]

        # Avoid division by zero (just in case)
        mask_sum = mask.sum(1).clamp(min=1e-9)             # [B, 1]

        pooled = (token_embeddings * mask).sum(1) / mask_sum
        return pooled  # [B, H]

    def embed_text_mean(self, text: str) -> torch.Tensor:
        """
        Compute embedding for a single text string.
        Returns a 1D CPU tensor of shape [H].
        """

        # Tokenize → send only input tensors to device
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Encode
        with torch.inference_mode():
            outputs = self.model(**inputs)

        pooled = self.mean_pooling(outputs, inputs["attention_mask"])  # [1, H]

        return pooled.squeeze(0).cpu()  # [H]
