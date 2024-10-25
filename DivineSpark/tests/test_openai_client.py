# DivineSpark/tests/test_openai_client.py

import unittest
from unittest.mock import patch, Mock
from DivineSpark.apis.openai_client import OpenAIClient
import os
import requests  # Importing requests to handle exceptions

class TestOpenAIClient(unittest.TestCase):
    def setUp(self):
        """
        Initializes the OpenAIClient with test API key and base URL.
        """
        # Load API key from environment variable
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = 'https://api.openai.com'  # Correct base URL without '/v1'
        if not self.api_key:
            self.fail("OPENAI_API_KEY environment variable not set.")
        
        self.client = OpenAIClient(api_key=self.api_key, base_url=self.base_url)

    @patch('DivineSpark.apis.openai_client.requests.post')
    def test_generate_completion(self, mock_post):
        """
        Tests the generate_completion method for successful response.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Test completion"}}]}
        mock_post.return_value = mock_response

        prompt = "Hello, world!"
        model_name = "gpt-4o"
        max_tokens = 10

        result = self.client.generate_completion(prompt, model_name, max_tokens)
        self.assertIn("choices", result)
        self.assertEqual(result["choices"][0]["message"]["content"], "Test completion")
        mock_post.assert_called_once()

    @patch('DivineSpark.apis.openai_client.requests.post')
    def test_generate_image(self, mock_post):
        """
        Tests the generate_image method for successful response.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"url": "https://example.com/image.png"}]}
        mock_post.return_value = mock_response

        prompt = "A beautiful landscape."
        model_name = "dall-e-3"

        result = self.client.generate_image(prompt, model_name)
        self.assertIn("data", result)
        self.assertEqual(result["data"][0]["url"], "https://example.com/image.png")
        mock_post.assert_called_once()

    @patch('DivineSpark.apis.openai_client.requests.post')
    def test_transcribe_audio(self, mock_post):
        """
        Tests the transcribe_audio method for successful response.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Transcribed text"}
        mock_post.return_value = mock_response

        audio_file = 'path/to/test_audio.wav'
        model_name = "whisper-1"

        # Mock opening the file using patch
        with patch('builtins.open', unittest.mock.mock_open(read_data=b'data')):
            result = self.client.transcribe_audio(audio_file, model_name)
            self.assertIn("text", result)
            self.assertEqual(result["text"], "Transcribed text")
            mock_post.assert_called_once()

    @patch('DivineSpark.apis.openai_client.requests.post')
    def test_handle_unauthorized_error(self, mock_post):
        """
        Tests handling of unauthorized (401) error response.
        """
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Client Error: Unauthorized for url")
        mock_post.return_value = mock_response

        prompt = "Hello, world!"
        model_name = "gpt-4o"
        max_tokens = 10

        with self.assertRaises(requests.exceptions.HTTPError):
            self.client.generate_completion(prompt, model_name, max_tokens)
        mock_post.assert_called_once()

    def tearDown(self):
        """
        Clean up after tests.
        """
        pass  # No cleanup necessary for mocked tests

if __name__ == '__main__':
    unittest.main()
