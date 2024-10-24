import os
import unittest
from unittest.mock import patch
from DivineSpark.core.divine_spark import DivineSpark
from DivineSpark.core.utils import Utils


class TestDivineSpark(unittest.TestCase):
    def setUp(self):
        # Retrieve the API key from environment variables
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API Key. Please set the OPENAI_API_KEY environment variable.")
        
        self.api_type = "openai"
        self.spark = DivineSpark(self.api_key, self.api_type)

        # Load OpenAI model configuration from file
        self.config_file = 'DivineSpark/config/models_config.json'
        if Utils.file_exists(self.config_file):
            self.models_config = Utils.load_from_json(self.config_file).get("openai", {})
        else:
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")

        # Register models
        required_models = {
            "gpt-4o": {
                "description": "Test model",
                "context_window": 128000,
                "max_output_tokens": 16384,
                "supports_text_completion": True
            },
            "dall-e-3": {
                "description": "Image generation model",
                "multimodal": True,
                "supports_image_generation": True
            },
            "whisper-1": {
                "description": "Speech-to-text model",
                "audio_input": True,
                "supports_audio_transcription": True
            }
        }

        for model_name, model_details in required_models.items():
            self.spark.register_model(model_name, model_details)

    @patch('requests.Session.post')
    def test_generate_text_completion(self, mock_post):
        # Mock a successful completion response
        mock_post.return_value.json.return_value = {
            "model": "gpt-4o-2024-08-06",
            "choices": [{"message": {"role": "assistant", "content": "Mocked response"}}]
        }
        mock_post.return_value.status_code = 200

        prompt = "Test prompt"
        result = self.spark.generate_text_completion(prompt, model_name="gpt-4o", max_tokens=10)
        self.assertIn("model", result)
        self.assertEqual(result["model"], "gpt-4o-2024-08-06")

    @patch('requests.Session.post')
    def test_generate_image(self, mock_post):
        # Mock a successful image generation response
        mock_post.return_value.json.return_value = {
            "created": 1729772286,
            "data": [
                {"url": "https://example.com/image.png"}
            ]
        }
        mock_post.return_value.status_code = 200

        prompt = "Generate a futuristic city skyline"
        result = self.spark.generate_image(prompt, model_name="dall-e-3")
        self.assertIn("data", result)
        self.assertGreater(len(result["data"]), 0)
        self.assertIn("url", result["data"][0])

    @patch('requests.Session.post')
    def test_transcribe_audio(self, mock_post):
        # Mock a successful audio transcription response
        mock_post.return_value.json.return_value = {
            "text": "Transcribed audio content"
        }
        mock_post.return_value.status_code = 200

        audio_model_name = "whisper-1"
        result = self.spark.transcribe_audio("test_audio_file.wav", model_name=audio_model_name)
        self.assertIn("text", result)
        self.assertEqual(result["text"], "Transcribed audio content")
