import json
from .api_client import BaseAPIClient
from .model_registry import ModelRegistry
from .model_loader import ModelLoader

class DivineSpark:
    def __init__(self, api_key, api_type):
        self.api_type = api_type.lower()
        self.api_client = self._initialize_api_client(api_key)
        self.model_registry = ModelRegistry()
        self.model_loader = ModelLoader(self.api_client, self.model_registry)

    def _initialize_api_client(self, api_key):
        if self.api_type == "openai":
            from ..apis.openai_client import OpenAIClient
            return OpenAIClient(api_key, base_url="https://api.openai.com/v1")
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")

    def register_model(self, model_name, model_details):
        self.model_registry.register_model(self.api_type, model_name, model_details)

    def use_model(self, model_name):
        model = self.model_loader.load_model(self.api_type, model_name)
        return model
