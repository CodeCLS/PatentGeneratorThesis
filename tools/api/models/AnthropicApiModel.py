# models/AnthropicApiModel.py
import anthropic


class AnthropicModel:
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", max_tokens: int = 200):
        self.client = anthropic.Anthropic()   # instantiate official client
        self.model = model
        self.max_tokens = max_tokens
        self.name = model

    def send(self, message: str) -> str:
        """Send a prompt to Anthropic Claude and return the text response."""
        print(f"[AnthropicModel] sending message to {self.name}: {message}")

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=(
                "You are a disciplined JSON generator. "
                "Always respond with a single JSON object and nothing else. "
                "No prose, no backticks, no explanations."
            ),
            messages=[{"role": "user", "content": message}],
        )

        # Anthropic SDK returns message parts under resp.content
        text = ""
        for block in getattr(resp, "content", []):
            if getattr(block, "type", None) == "text":
                text += getattr(block, "text", "")

        return text
