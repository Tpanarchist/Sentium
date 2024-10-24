import os
import unittest
from DivineSpark.apis.openai_client import OpenAIClient

class TestOpenAIClient(unittest.TestCase):
    def setUp(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API Key. Please set the OPENAI_API_KEY environment variable.")
        self.client = OpenAIClient(self.api_key)

    def test_generate_completion(self):
        prompt = "Test prompt"
        try:
            result = self.client.generate_completion(prompt, model_name="gpt-4o", max_tokens=10)
            # Check if the response has a 'model' field indicating the model used
            self.assertIn("model", result)
            self.assertEqual(result["model"], "gpt-4o-2024-08-06")

            # Check if the 'choices' field exists and contains the generated message
            self.assertIn("choices", result)
            self.assertIsInstance(result["choices"], list)
            self.assertGreater(len(result["choices"]), 0)
            self.assertIn("message", result["choices"][0])
            self.assertIn("content", result["choices"][0]["message"])
        except Exception as e:
            self.fail(f"Test failed due to an API error: {e}")

if __name__ == "__main__":
    unittest.main()
