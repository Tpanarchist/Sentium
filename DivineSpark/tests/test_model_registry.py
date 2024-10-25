# DivineSpark/tests/test_model_registry.py

import unittest
from DivineSpark.core.model_registry import ModelRegistry
import os

class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ModelRegistry()
        # Path to models_config.json relative to this test file
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'models_config.json')
        print(f"Loading models from configuration file at '{config_path}'.")
        self.registry.load_from_file(config_path)

    def test_filter_multimodal_models(self):
        print("Testing filter_multimodal_models.")
        multimodal_models = self.registry.list_models(filter_multimodal=True)
        print(f"Multimodal models: {multimodal_models}")

        # Based on models_config.json, 'gpt-4o' and 'gpt-4o-mini' are multimodal
        self.assertIn("gpt-4o-mini", multimodal_models["openai"], "'gpt-4o-mini' should be in multimodal models.")
        self.assertIn("gpt-4o", multimodal_models["openai"], "'gpt-4o' should be in multimodal models.")

    def test_filter_audio_models(self):
        print("Testing filter_audio_models.")
        audio_models = self.registry.list_models(filter_audio=True)
        print(f"Audio models: {audio_models}")

        # Based on models_config.json, 'whisper-1', 'tts-1', and 'tts-1-hd' are audio models
        self.assertIn("whisper-1", audio_models["openai"], "'whisper-1' should be in audio models.")
        self.assertIn("tts-1", audio_models["openai"], "'tts-1' should be in audio models.")
        self.assertIn("tts-1-hd", audio_models["openai"], "'tts-1-hd' should be in audio models.")

    def test_get_model_valid(self):
        print("Testing get_model with a valid model.")
        model = self.registry.get_model("openai", "gpt-4o")
        self.assertIsNotNone(model, "get_model returned None for a valid model.")

    def test_get_model_invalid(self):
        print("Testing get_model with an invalid model.")
        with self.assertRaises(ValueError):
            self.registry.get_model("openai", "non-existent-model")

    def tearDown(self):
        print("TestModelRegistry tearDown completed.")
