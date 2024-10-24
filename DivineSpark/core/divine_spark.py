import json
from .api_client import BaseAPIClient
from .model_registry import ModelRegistry
from .model_loader import ModelLoader

class DivineSpark:
    def __init__(self, api_key, api_type, config_file=None):
        self.api_type = api_type.lower()
        self.api_client = self._initialize_api_client(api_key)
        self.model_registry = ModelRegistry()
        self.model_loader = ModelLoader(self.api_client, self.model_registry)

        # Load model configuration from file if provided
        if config_file:
            self.load_models_from_file(config_file)

    def _initialize_api_client(self, api_key):
        if self.api_type == "openai":
            from ..apis.openai_client import OpenAIClient
            return OpenAIClient(api_key, base_url="https://api.openai.com/v1")
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")

    def register_model(self, model_name, model_details):
        # Register the model using the updated registry
        self.model_registry.register_model(self.api_type, model_name, model_details)

    def use_model(self, model_name):
        # Load the model and handle multimodal capabilities
        model = self.model_loader.load_model(self.api_type, model_name)
        model_details = self.model_registry.get_model(self.api_type, model_name)

        if model_details.get("multimodal"):
            print(f"Multimodal model '{model_name}' is loaded and ready.")
        
        return model

    def load_models_from_file(self, config_file):
        # Load model configurations from a JSON file
        try:
            self.model_registry.load_from_file(config_file)
        except Exception as e:
            print(f"Error loading models from file: {e}")

    def generate_text_completion(self, prompt, model_name="gpt-4o", max_tokens=50):
        # Use a text-based model to generate a completion
        model_details = self.model_registry.get_model(self.api_type, model_name)
        if not model_details:
            raise ValueError(f"Model {model_name} is not registered.")
        
        if model_details.get("endpoint") != "chat/completions":
            raise ValueError(f"Model {model_name} is not compatible with text completions.")
        
        return self.api_client.generate_completion(prompt, model_name, max_tokens)

    def generate_image(self, prompt, model_name="dall-e-3"):
        # Use an image generation model to create an image
        model_details = self.model_registry.get_model(self.api_type, model_name)
        if not model_details or model_details.get("endpoint") != "images/generations":
            raise ValueError(f"Model {model_name} is not compatible with image generation.")
        
        return self.api_client.generate_image(prompt, model_name)

    def transcribe_audio(self, audio_file, model_name="whisper-1"):
        # Use an audio model to transcribe audio
        model_details = self.model_registry.get_model(self.api_type, model_name)
        if not model_details or model_details.get("endpoint") != "audio/transcriptions":
            raise ValueError(f"Model {model_name} is not compatible with audio transcription.")
        
        return self.api_client.transcribe_audio(audio_file, model_name)
