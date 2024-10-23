import unittest
from ..apis.openai_client import OpenAIClient

class TestOpenAIClient(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.client = OpenAIClient(self.api_key)

    def test_generate_completion(self):
        prompt = "Test prompt"
        result = self.client.generate_completion(prompt, model_name="gpt-4o", max_tokens=10)
        self.assertIn("model_name", result)
        self.assertEqual(result["model_name"], "gpt-4o")

if __name__ == "__main__":
    unittest.main()
