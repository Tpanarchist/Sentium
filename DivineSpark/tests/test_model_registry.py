import unittest
from DivineSpark.core.model_registry import ModelRegistry
import json
import os

class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        # Initialize a fresh instance of ModelRegistry for each test
        self.registry = ModelRegistry()

    def test_register_and_retrieve_model(self):
        # Test the basic registration and retrieval of a model
        api_name = "openai"
        model_name = "gpt-4o"
        model_details = {"description": "Test model", "multimodal": True}
        self.registry.register_model(api_name, model_name, model_details)
        retrieved_model = self.registry.get_model(api_name, model_name)
        self.assertEqual(retrieved_model["description"], "Test model")
        self.assertTrue(retrieved_model["multimodal"])

    def test_list_models(self):
        # Test listing models for a specific API
        api_name = "openai"
        self.registry.register_model(api_name, "gpt-4o", {"description": "Test model"})
        self.registry.register_model(api_name, "gpt-4o-mini", {"description": "Another test model"})
        models = self.registry.list_models(api_name)
        self.assertIn("gpt-4o", models)
        self.assertIn("gpt-4o-mini", models)

    def test_filter_multimodal_models(self):
        # Test filtering models based on multimodal capabilities
        api_name = "openai"
        self.registry.register_model(api_name, "gpt-4o", {"description": "Test model", "multimodal": True})
        self.registry.register_model(api_name, "gpt-4o-mini", {"description": "Non-multimodal model", "multimodal": False})
        multimodal_models = self.registry.list_models(api_name, filter_multimodal=True)
        non_multimodal_models = self.registry.list_models(api_name, filter_multimodal=False)
        
        self.assertIn("gpt-4o", multimodal_models)
        self.assertNotIn("gpt-4o-mini", multimodal_models)
        self.assertIn("gpt-4o-mini", non_multimodal_models)
        self.assertNotIn("gpt-4o", non_multimodal_models)

    def test_load_models_from_file(self):
        # Test loading models from a configuration file
        test_config_file = 'DivineSpark/config/test_models_config.json'
        test_data = {
            "openai": {
                "test-model": {
                    "description": "Loaded from file",
                    "context_window": 1000,
                    "max_output_tokens": 200
                }
            }
        }
        # Create a test config file
        with open(test_config_file, 'w') as file:
            json.dump(test_data, file)

        # Load models from the file
        self.registry.load_from_file(test_config_file)
        loaded_model = self.registry.get_model("openai", "test-model")
        
        # Assertions to verify correct loading
        self.assertEqual(loaded_model["description"], "Loaded from file")
        self.assertEqual(loaded_model["max_output_tokens"], 200)

        # Clean up test file
        os.remove(test_config_file)

    def test_save_to_file(self):
        # Test saving models to a configuration file
        api_name = "openai"
        model_name = "gpt-4o"
        model_details = {"description": "Test model", "context_window": 1000}
        
        # Register and save the model
        self.registry.register_model(api_name, model_name, model_details)
        save_file_path = 'DivineSpark/config/saved_models_config.json'
        self.registry.save_to_file(save_file_path)
        
        # Load the saved file to verify its content
        with open(save_file_path, 'r') as file:
            saved_data = json.load(file)
        
        # Assertions to check if the data was saved correctly
        self.assertIn("openai", saved_data)
        self.assertIn("gpt-4o", saved_data["openai"])
        self.assertEqual(saved_data["openai"]["gpt-4o"]["description"], "Test model")
        
        # Clean up test file
        os.remove(save_file_path)

    def test_get_nonexistent_model(self):
        # Test retrieval of a non-existent model to ensure proper error handling
        with self.assertRaises(ValueError):
            self.registry.get_model("openai", "nonexistent-model")

if __name__ == "__main__":
    unittest.main()
