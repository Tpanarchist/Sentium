# File: DivineSpark/apis/openai_client.py

from ..core.api_client import BaseAPIClient

class OpenAIClient(BaseAPIClient):
    def __init__(self, api_key, base_url="https://api.openai.com/v1"):
        super().__init__(api_key, base_url)

    def load_model(self, model_name, **kwargs):
        print(f"Loading OpenAI model: {model_name}")
        return {"model_name": model_name, **kwargs}
    
    def generate_completion(self, prompt, model_name="gpt-4o", max_tokens=50):
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        # Use chat/completions for models that support conversation-based interactions
        return self.make_request("chat/completions", payload)
