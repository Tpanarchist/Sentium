import unittest
from ..core.divine_spark import DivineSpark

class TestDivineSpark(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.api_type = "openai"
        self.spark = DivineSpark(self.api_key, self.api_type)

    def test_register_model(self):
        model_name = "gpt-4o"
        model_details = {
            "description": "Test model",
            "context_window": 128000,
            "max_output_tokens": 16384
        }
        self.spark.register_model(model_name, model_details)
        registered_model = self.spark.model_registry.get_model(self.api_type, model_name)
        self.assertEqual(registered_model["description"], "Test model")

    def test_use_model(self):
        model_name = "gpt-4o"
        model_details = {
            "description": "Test model",
            "context_window": 128000,
            "max_output_tokens": 16384
        }
        self.spark.register_model(model_name, model_details)
        loaded_model = self.spark.use_model(model_name)
        self.assertEqual(loaded_model["model_name"], "gpt-4o")

if __name__ == "__main__":
    unittest.main()
