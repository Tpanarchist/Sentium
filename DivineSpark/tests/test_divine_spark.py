# DivineSpark/tests/test_divine_spark.py

import unittest
from unittest.mock import patch, Mock
from DivineSpark.core.divine_spark import DivineSpark
import os
import requests  # Importing requests to handle exceptions

class TestDivineSpark(unittest.TestCase):
    def setUp(self):
        """
        Initializes the DivineSpark instance with test API key and configuration file.
        """
        # Load API key from environment variable
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            self.fail("OPENAI_API_KEY environment variable not set.")
        
        # Path to models_config.json relative to this test file
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'models_config.json')
        print(f"Loading models from configuration file at '{config_path}'.")
        self.spark = DivineSpark(api_key=self.api_key, api_type='openai', config_file=config_path)

    @patch('DivineSpark.apis.openai_client.requests.post')
    def test_generate_image(self, mock_post):
        """
        Tests the generate_image method for successful image generation.
        """
        prompt = "A sunset over mountains."
        print(f"Testing generate_image with prompt: '{prompt}'.")
        
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"url": "https://example.com/image.png"}]}
        mock_post.return_value = mock_response

        try:
            result = self.spark.generate_image(prompt, model_name="dall-e-3")
            self.assertIsNotNone(result, "generate_image returned None.")
            self.assertIn("data", result)
            self.assertEqual(result["data"][0]["url"], "https://example.com/image.png")
            print("generate_image test passed.")
        except ValueError as e:
            self.fail(f"generate_image raised ValueError unexpectedly: {e}")
        except requests.exceptions.HTTPError as e:
            self.fail(f"generate_image raised HTTPError: {e}")

    @patch('DivineSpark.apis.openai_client.requests.post')
    def test_generate_text_completion(self, mock_post):
        """
        Tests the generate_text_completion method for successful text generation.
        """
        prompt = "Once upon a time"
        print(f"Testing generate_text_completion with prompt: '{prompt}'.")
        
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Test completion"}}]}
        mock_post.return_value = mock_response

        try:
            result = self.spark.generate_text_completion(prompt, model_name="gpt-4o", max_tokens=10)
            self.assertIsNotNone(result, "generate_text_completion returned None.")
            self.assertIn("choices", result)
            self.assertEqual(result["choices"][0]["message"]["content"], "Test completion")
            print("generate_text_completion test passed.")
        except ValueError as e:
            self.fail(f"generate_text_completion raised ValueError unexpectedly: {e}")
        except requests.exceptions.HTTPError as e:
            self.fail(f"generate_text_completion raised HTTPError: {e}")

    @patch('DivineSpark.apis.openai_client.requests.post')
    def test_transcribe_audio(self, mock_post):
        """
        Tests the transcribe_audio method for successful audio transcription.
        """
        audio_model_name = "whisper-1"
        # Path to the dummy audio file
        audio_file_path = os.path.join(os.path.dirname(__file__), 'test_audio_file.wav')
        print(f"Testing transcribe_audio with audio file: '{audio_file_path}'.")

        # Create a dummy audio file for testing purposes if it doesn't exist
        if not os.path.exists(audio_file_path):
            with open(audio_file_path, 'wb') as f:
                f.write(b'\x00\x00\x00')  # Dummy content
            print(f"Created dummy audio file at '{audio_file_path}' for testing.")

        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Transcribed text"}
        mock_post.return_value = mock_response

        try:
            result = self.spark.transcribe_audio(audio_file_path, model_name=audio_model_name)
            self.assertIsNotNone(result, "transcribe_audio returned None.")
            self.assertIn("text", result)
            self.assertEqual(result["text"], "Transcribed text")
            print("transcribe_audio test passed.")
        except ValueError as e:
            self.fail(f"transcribe_audio raised ValueError unexpectedly: {e}")
        except requests.exceptions.HTTPError as e:
            self.fail(f"transcribe_audio raised HTTPError: {e}")

    @patch('DivineSpark.apis.openai_client.requests.post')
    def test_handle_unauthorized_error(self, mock_post):
        """
        Tests handling of unauthorized (401) error response.
        """
        prompt = "Hello, world!"
        model_name = "gpt-4o"
        max_tokens = 10

        # Mock API response to simulate unauthorized error
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Client Error: Unauthorized for url")
        mock_post.return_value = mock_response

        with self.assertRaises(requests.exceptions.HTTPError):
            self.spark.generate_text_completion(prompt, model_name, max_tokens)
        mock_post.assert_called_once()

    def tearDown(self):
        """
        Cleans up the dummy audio file after tests to prevent PermissionError.
        """
        # Path to the dummy audio file
        audio_file_path = os.path.join(os.path.dirname(__file__), 'test_audio_file.wav')
        if os.path.exists(audio_file_path):
            try:
                os.remove(audio_file_path)
                print(f"Removed dummy audio file at '{audio_file_path}' after testing.")
            except PermissionError as e:
                print(f"PermissionError while trying to remove '{audio_file_path}': {e}")

if __name__ == '__main__':
    unittest.main()
