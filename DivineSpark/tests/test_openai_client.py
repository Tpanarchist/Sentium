import os
import unittest
from DivineSpark.apis.openai_client import OpenAIClient

class TestOpenAIClient(unittest.TestCase):
    def setUp(self):
        # Load API key from environment variable
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API Key. Please set the OPENAI_API_KEY environment variable.")
        
        # Initialize the OpenAI client
        self.client = OpenAIClient(self.api_key)

    def test_generate_completion(self):
        # Test generating a simple text completion
        prompt = "Test prompt"
        try:
            result = self.client.generate_completion(prompt, model_name="gpt-4o", max_tokens=10)
            
            # Verify response structure and content
            self.assertIn("model", result)
            self.assertEqual(result["model"], "gpt-4o-2024-08-06")
            self.assertIn("choices", result)
            self.assertIsInstance(result["choices"], list)
            self.assertGreater(len(result["choices"]), 0)
            self.assertIn("message", result["choices"][0])
            self.assertIn("content", result["choices"][0]["message"])
        except Exception as e:
            self.fail(f"Test failed due to an API error: {e}")

    def test_generate_completion_invalid_model(self):
        # Test with an invalid model name to ensure proper error handling
        prompt = "Test prompt with invalid model"
        with self.assertRaises(Exception):
            self.client.generate_completion(prompt, model_name="invalid-model", max_tokens=10)

    def test_generate_image(self):
        # Test generating an image using the DALL-E model
        prompt = "Generate a futuristic city skyline at night"
        try:
            result = self.client.generate_image(prompt, model_name="dall-e-3")
            
            # Verify response structure for image generation
            self.assertIn("image_url", result)
            self.assertIsInstance(result["image_url"], str)
            self.assertTrue(result["image_url"].startswith("http"))
        except Exception as e:
            self.fail(f"Image generation test failed due to an API error: {e}")

    def test_generate_image_invalid_model(self):
        # Test generating an image using an invalid model
        prompt = "Test prompt for invalid image model"
        with self.assertRaises(Exception):
            self.client.generate_image(prompt, model_name="invalid-image-model")

    def test_transcribe_audio(self):
        # Test transcribing an audio file using Whisper
        # Note: Replace 'test_audio_file.wav' with a valid audio file path for an actual test
        audio_file_path = "test_audio_file.wav"
        try:
            result = self.client.transcribe_audio(audio_file_path, model_name="whisper-1")
            
            # Verify response structure for audio transcription
            self.assertIn("transcription", result)
            self.assertIsInstance(result["transcription"], str)
        except FileNotFoundError:
            self.skipTest("Test audio file not found. Skipping transcription test.")
        except Exception as e:
            self.fail(f"Audio transcription test failed due to an API error: {e}")

    def test_handle_unauthorized_error(self):
        # Test handling an unauthorized request due to invalid API key
        invalid_client = OpenAIClient("invalid_key")
        prompt = "Test unauthorized access"
        with self.assertRaises(Exception) as context:
            invalid_client.generate_completion(prompt, model_name="gpt-4o", max_tokens=10)
        self.assertIn("Unauthorized", str(context.exception))

if __name__ == "__main__":
    unittest.main()
