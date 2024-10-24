import os
import unittest
from unittest.mock import patch
from DivineSpark.apis.openai_client import OpenAIClient

class TestOpenAIClient(unittest.TestCase):
    def setUp(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API Key. Please set the OPENAI_API_KEY environment variable.")
        self.client = OpenAIClient(self.api_key)

    @patch('requests.Session.post')
    def test_generate_completion(self, mock_post):
        # Mock the response for a successful text completion
        mock_post.return_value.json.return_value = {
            "model": "gpt-4o-2024-08-06",
            "choices": [{"message": {"role": "assistant", "content": "Mocked response"}}]
        }
        mock_post.return_value.status_code = 200

        prompt = "Test prompt"
        result = self.client.generate_completion(prompt, model_name="gpt-4o", max_tokens=10)
        self.assertIn("model", result)
        self.assertEqual(result["model"], "gpt-4o-2024-08-06")

    @patch('requests.Session.post')
    def test_generate_image(self, mock_post):
        # Mock the response for image generation
        mock_post.return_value.json.return_value = {
            "created": 1729772286,
            "data": [
                {"url": "https://example.com/image.png"}
            ]
        }
        mock_post.return_value.status_code = 200

        prompt = "Generate a futuristic city skyline"
        result = self.client.generate_image(prompt, model_name="dall-e-3")
        self.assertIn("data", result)
        self.assertGreater(len(result["data"]), 0)
        self.assertIn("url", result["data"][0])

    @patch('requests.Session.post')
    def test_handle_unauthorized_error(self, mock_post):
        # Mock an unauthorized response
        mock_post.return_value.status_code = 401
        mock_post.return_value.json.return_value = {
            "error": {
                "message": "Incorrect API key provided.",
                "type": "invalid_request_error"
            }
        }

        with self.assertRaises(Exception) as context:
            prompt = "Test prompt"
            self.client.generate_completion(prompt, model_name="gpt-4o", max_tokens=10)

        self.assertIn("Unauthorized", str(context.exception))
