# DivineSpark/core/divine_spark.py

from .api_client import BaseAPIClient
from .model_registry import ModelRegistry
from .model_loader import ModelLoader
import os  # Importing os to access environment variables

class DivineSpark:
    def __init__(self, api_key=None, api_type="openai", config_file=None):
        """
        Initializes the DivineSpark instance with the given API key and type.

        Args:
            api_key (str, optional): Your API key for the specified API.
                                      If not provided, it will attempt to load from environment variables.
            api_type (str, optional): The type of API to use (e.g., "openai").
            config_file (str, optional): Path to the models configuration JSON file.
        """
        self.api_type = api_type.lower()

        # Load API key from parameter or environment variable
        if api_key is None:
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("API key not provided and 'OPENAI_API_KEY' environment variable not set.")
        else:
            self.api_key = api_key

        self.api_client = self._initialize_api_client(self.api_key)
        self.model_registry = ModelRegistry()
        self.model_loader = ModelLoader(self.api_client, self.model_registry)

        # Load model configuration from file if provided
        if config_file:
            self.load_models_from_file(config_file)

    def _initialize_api_client(self, api_key):
        """
        Initializes the API client based on the API type.

        Args:
            api_key (str): Your API key.

        Returns:
            BaseAPIClient: An instance of the appropriate API client.

        Raises:
            ValueError: If the API type is unsupported.
        """
        if self.api_type == "openai":
            from ..apis.openai_client import OpenAIClient
            print("Initializing OpenAIClient.")
            return OpenAIClient(api_key, base_url="https://api.openai.com")
        else:
            raise ValueError(f"Unsupported API type: '{self.api_type}'.")

    def register_model(self, model_name, model_details):
        """
        Registers a new model with the model registry.

        Args:
            model_name (str): The name of the model to register.
            model_details (dict): The details of the model.
        """
        self.model_registry.register_model(self.api_type, model_name, model_details)
        print(f"Model '{model_name}' registered successfully.")

    def use_model(self, model_name):
        """
        Loads and prepares a model for use, handling multimodal capabilities.

        Args:
            model_name (str): The name of the model to use.

        Returns:
            dict: The loaded model details.

        Raises:
            ValueError: If the model cannot be found or is incompatible.
        """
        print(f"Attempting to load model '{model_name}'.")
        model = self.model_loader.load_model(self.api_type, model_name)
        model_details = self.model_registry.get_model(self.api_type, model_name)

        if model_details.get("capabilities", {}).get("output") == ["text"] and "image" in model_details.get("capabilities", {}).get("input", []):
            print(f"Multimodal model '{model_name}' is loaded and ready for text and image inputs.")

        elif model_details.get("capabilities", {}).get("input") == ["audio"] and model_details.get("capabilities", {}).get("output") == ["text"]:
            print(f"Audio model '{model_name}' is loaded and ready for audio to text transcription.")

        return model

    def load_models_from_file(self, config_file):
        """
        Loads model configurations from a JSON file into the model registry.

        Args:
            config_file (str): The path to the models configuration JSON file.
        """
        try:
            self.model_registry.load_from_file(config_file)
            loaded_models = list(self.model_registry.models[self.api_type].keys())
            print(f"Loaded models: {loaded_models}")
        except Exception as e:
            print(f"Error loading models from file: {e}")

    def generate_text_completion(self, prompt, model_name="gpt-4o", max_tokens=50):
        """
        Generates a text completion using the specified model.

        Args:
            prompt (str): The input prompt for text generation.
            model_name (str): The name of the model to use.
            max_tokens (int): The maximum number of tokens to generate.

        Returns:
            dict: The JSON response from the API.

        Raises:
            ValueError: If the model is not registered or incompatible.
        """
        print(f"Generating text completion using model '{model_name}'.")
        model_details = self.model_registry.get_model(self.api_type, model_name)
        if not model_details:
            raise ValueError(f"Model '{model_name}' is not registered.")

        endpoint = self.model_loader.get_endpoint(self.api_type, model_name)
        if endpoint != "chat/completions":
            raise ValueError(f"Model '{model_name}' is not compatible with text completions.")

        return self.api_client.generate_completion(prompt, model_name, max_tokens)

    def generate_image(self, prompt, model_name="dall-e-3"):
        """
        Generates an image based on the provided prompt using the specified model.

        Args:
            prompt (str): The input prompt for image generation.
            model_name (str): The name of the image generation model to use.

        Returns:
            dict: The JSON response from the API.

        Raises:
            ValueError: If the model is not registered or incompatible.
        """
        print(f"Generating image using model '{model_name}'.")
        model_details = self.model_registry.get_model(self.api_type, model_name)
        if not model_details or model_details.get("endpoint") != "images/generations":
            raise ValueError(f"Model '{model_name}' is not compatible with image generation.")

        return self.api_client.generate_image(prompt, model_name)

    def transcribe_audio(self, audio_file, model_name="whisper-1"):
        """
        Transcribes the provided audio file using the specified model.

        Args:
            audio_file (str): The path to the audio file to transcribe.
            model_name (str): The name of the audio transcription model to use.

        Returns:
            dict: The JSON response from the API.

        Raises:
            ValueError: If the model is not registered or incompatible.
        """
        print(f"Transcribing audio using model '{model_name}'.")
        model_details = self.model_registry.get_model(self.api_type, model_name)
        if not model_details or model_details.get("endpoint") != "audio/transcriptions":
            raise ValueError(f"Model '{model_name}' is not compatible with audio transcription.")

        return self.api_client.transcribe_audio(audio_file, model_name)
