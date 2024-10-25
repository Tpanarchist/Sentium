# DivineSpark/tests/test_model_registry.py

import unittest
from unittest.mock import patch, Mock
from DivineSpark.core.model_registry import ModelRegistry
import os

class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        """
        Initializes the ModelRegistry instance and loads models from the configuration file.
        """
        self.registry = ModelRegistry()
        # Path to models_config.json relative to this test file
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'models_config.json')
        print(f"Loading models from configuration file at '{config_path}'.")
        self.registry.load_from_file(config_path)
        print("Registered Models:")
        self.registry.list_all_models()

    def test_filter_audio_models(self):
        """
        Tests filtering of audio models.
        """
        print("Testing filter_audio_models.")
        audio_models = self.registry.list_models(filter_audio=True)
        expected_keys = {'tts-1', 'tts-1-hd', 'whisper-1'}
        actual_keys = set(audio_models.get('openai', {}).keys())
        self.assertEqual(actual_keys, expected_keys)
        print(f"Audio models: {audio_models.get('openai')}")

    def test_filter_multimodal_models(self):
        """
        Tests filtering of multimodal models.
        """
        print("Testing filter_multimodal_models.")
        multimodal_models = self.registry.list_models(filter_multimodal=True)
        expected_keys = {'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'dall-e-3', 'dall-e-2'}
        actual_keys = set(multimodal_models.get('openai', {}).keys())
        self.assertEqual(actual_keys, expected_keys)
        print(f"Multimodal models: {multimodal_models.get('openai')}")

    def test_get_model_invalid(self):
        """
        Tests retrieving a non-existent model.
        """
        print("Testing get_model with an invalid model.")
        with self.assertRaises(ValueError):
            self.registry.get_model('openai', 'non-existent-model')

    def test_get_model_valid(self):
        """
        Tests retrieving a valid model.
        """
        print("Testing get_model with a valid model.")
        model = self.registry.get_model('openai', 'gpt-4o')
        self.assertIsNotNone(model)
        self.assertEqual(model['description'], "Our high-intelligence flagship model for complex, multi‑step tasks.")
        print(f"Retrieved Model: {model}")

    def tearDown(self):
        """
        Clean up after tests.
        """
        print("TestModelRegistry tearDown completed.")

if __name__ == '__main__':
    unittest.main()
