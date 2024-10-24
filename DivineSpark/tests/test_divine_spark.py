import unittest
import json
from DivineSpark.core.divine_spark import DivineSpark


class TestDivineSpark(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.api_type = "openai"
        self.spark = DivineSpark(self.api_key, self.api_type)

        # Mock model registration for testing purposes
        self.test_model_name = "gpt-4o"
        self.test_model_details = {
            "description": "Test model",
            "context_window": 128000,
            "max_output_tokens": 16384,
            "multimodal": True,
            "endpoint": "chat/completions"
        }
        self.spark.register_model(self.test_model_name, self.test_model_details)

    def test_register_model(self):
        # Verify the model registration process
        registered_model = self.spark.model_registry.get_model(self.api_type, self.test_model_name)
        self.assertEqual(registered_model["description"], "Test model")
        self.assertEqual(registered_model["max_output_tokens"], 16384)

    def test_use_model(self):
        # Verify that a registered model can be used
        loaded_model = self.spark.use_model(self.test_model_name)
        self.assertEqual(loaded_model["model_name"], self.test_model_name)
        self.assertTrue(loaded_model["model_name"] in self.spark.model_registry.list_models(self.api_type))

    def test_generate_text_completion(self):
        # Test generating a text completion using a valid model
        prompt = "Hello, world!"
        result = self.spark.generate_text_completion(prompt, model_name="gpt-4o", max_tokens=10)
        self.assertIsNotNone(result)

    def test_generate_text_completion_invalid_model(self):
        # Test generating a text completion using an invalid model
        with self.assertRaises(ValueError):
            self.spark.generate_text_completion("Test prompt", model_name="invalid-model")

    def test_generate_image(self):
        # Mock registration for an image generation model
        image_model_name = "dall-e-3"
        image_model_details = {
            "description": "Image generation model",
            "multimodal": True,
            "endpoint": "images/generations"
        }
        self.spark.register_model(image_model_name, image_model_details)

        # Test generating an image using the valid model
        prompt = "Generate a sunset over mountains"
        result = self.spark.generate_image(prompt, model_name=image_model_name)
        self.assertIsNotNone(result)

    def test_generate_image_invalid_model(self):
        # Test generating an image using an invalid model
        with self.assertRaises(ValueError):
            self.spark.generate_image("Test prompt", model_name="invalid-image-model")

    def test_transcribe_audio(self):
        # Mock registration for an audio transcription model
        audio_model_name = "whisper-1"
        audio_model_details = {
            "description": "Audio transcription model",
            "audio_input": True,
            "endpoint": "audio/transcriptions"
        }
        self.spark.register_model(audio_model_name, audio_model_details)

        # Test transcribing an audio file
        # Note: For a real test, replace 'test_audio_file.wav' with a valid audio file path
        result = self.spark.transcribe_audio("test_audio_file.wav", model_name=audio_model_name)
        self.assertIsNotNone(result)

    def test_load_models_from_file(self):
        # Test loading models from a JSON configuration file
        test_config_file = 'DivineSpark/config/test_models_config.json'
        with open(test_config_file, 'w') as file:
            test_config_data = {
                "openai": {
                    "test-model": {
                        "description": "Loaded from file",
                        "context_window": 1000,
                        "max_output_tokens": 200
                    }
                }
            }
            json.dump(test_config_data, file)

        # Load models from the file
        self.spark.load_models_from_file(test_config_file)

        # Verify that the model was loaded correctly
        loaded_model = self.spark.model_registry.get_model("openai", "test-model")
        self.assertEqual(loaded_model["description"], "Loaded from file")
        self.assertEqual(loaded_model["max_output_tokens"], 200)


if __name__ == "__main__":
    unittest.main()
