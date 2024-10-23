import unittest
from ..core.model_registry import ModelRegistry

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

if __name__ == "__main__":
    unittest.main()
