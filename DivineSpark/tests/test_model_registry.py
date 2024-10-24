import os
import json  # Required for handling JSON data
import unittest
from DivineSpark.core.model_registry import ModelRegistry
from DivineSpark.core.utils import Utils

def create_test_config_file(data, file_path):
    """Utility to create a test configuration file."""
    with open(file_path, 'w') as file:
        json.dump(data, file)

class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ModelRegistry()

    def test_register_and_retrieve_model(self):
        api_name = "openai"
        model_name = "gpt-4o"
        model_details = {"description": "Test model"}
        self.registry.register_model(api_name, model_name, model_details)
        retrieved_model = self.registry.get_model(api_name, model_name)
        self.assertEqual(retrieved_model["description"], "Test model")

    def test_list_models(self):
        api_name = "openai"
        self.registry.register_model(api_name, "gpt-4o", {"description": "Test model"})
        models = self.registry.list_models(api_name)
        self.assertIn("gpt-4o", models)

    def test_load_models_from_file(self):
        # Test loading models from a JSON config file
        test_data = {
            "openai": {
                "test-model": {
                    "description": "Loaded from file",
                    "context_window": 1000,
                    "max_output_tokens": 200
                }
            }
        }
        test_file_path = 'DivineSpark/config/test_models_config.json'
        create_test_config_file(test_data, test_file_path)

        try:
            self.registry.load_from_file(test_file_path)
            loaded_model = self.registry.get_model("openai", "test-model")
            self.assertEqual(loaded_model["description"], "Loaded from file")
        finally:
            os.remove(test_file_path)

    def test_filter_multimodal_models(self):
        api_name = "openai"
        self.registry.register_model(api_name, "gpt-4o", {"description": "Test model", "multimodal": True})
        self.registry.register_model(api_name, "gpt-4o-mini", {"description": "Test model", "multimodal": False})

        multimodal_models = self.registry.list_models(api_name, filter_multimodal=True)
        self.assertIn("gpt-4o", multimodal_models)
        self.assertNotIn("gpt-4o-mini", multimodal_models)
