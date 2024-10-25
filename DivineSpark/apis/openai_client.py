# DivineSpark/apis/openai_client.py

import requests

class OpenAIClient:
    def __init__(self, api_key, base_url):
        """
        Initializes the OpenAIClient with the provided API key and base URL.

        Args:
            api_key (str): Your OpenAI API key.
            base_url (str): The base URL for the OpenAI API (e.g., "https://api.openai.com").
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')  # Ensure no trailing slash

    def generate_completion(self, prompt, model_name, max_tokens):
        """
        Generates a text completion using the specified model.

        Args:
            prompt (str): The input prompt for text generation.
            model_name (str): The name of the model to use.
            max_tokens (int): The maximum number of tokens to generate.

        Returns:
            dict: The JSON response from the API.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        endpoint = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        print(f"API Response for generate_completion: {response.json()}")
        return response.json()

    def generate_image(self, prompt, model_name):
        """
        Generates an image based on the provided prompt using the specified model.

        Args:
            prompt (str): The input prompt for image generation.
            model_name (str): The name of the image generation model to use.

        Returns:
            dict: The JSON response from the API.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        endpoint = f"{self.base_url}/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "prompt": prompt
        }
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        print(f"API Response for generate_image: {response.json()}")
        return response.json()

    def transcribe_audio(self, audio_file, model_name):
        """
        Transcribes the provided audio file using the specified model.

        Args:
            audio_file (str): The path to the audio file to transcribe.
            model_name (str): The name of the audio transcription model to use.

        Returns:
            dict: The JSON response from the API.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        endpoint = f"{self.base_url}/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
            # 'Content-Type' is handled by requests when using 'files'
        }
        data = {
            "model": model_name
        }
        # Use a context manager to ensure the file is closed after the request
        with open(audio_file, 'rb') as f:
            files = {
                'file': f
            }
            response = requests.post(endpoint, headers=headers, files=files, data=data)
            response.raise_for_status()
            print(f"API Response for transcribe_audio: {response.json()}")
            return response.json()
