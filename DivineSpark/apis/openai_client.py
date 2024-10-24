from ..core.api_client import BaseAPIClient

class OpenAIClient(BaseAPIClient):
    def __init__(self, api_key, base_url="https://api.openai.com/v1"):
        super().__init__(api_key, base_url)

    def load_model(self, model_name, **kwargs):
        print(f"Loading OpenAI model: {model_name}")
        return {"model_name": model_name, **kwargs}

    def generate_completion(self, prompt, model_name="gpt-4o", max_tokens=50):
        # Use chat/completions for models that support conversation-based interactions
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        return self.make_request("chat/completions", payload)

    def generate_image(self, prompt, model_name="dall-e-3"):
        # Use images/generations for models that support image generation
        payload = {
            "model": model_name,
            "prompt": prompt
        }
        return self.make_request("images/generations", payload)

    def transcribe_audio(self, audio_file_path, model_name="whisper-1"):
        # Use audio/transcriptions for models that support audio transcription
        with open(audio_file_path, "rb") as audio_file:
            files = {
                'file': (audio_file_path, audio_file, 'audio/wav')
            }
            data = {
                "model": model_name
            }
            return self.make_request("audio/transcriptions", data=data, files=files)

    def make_request(self, endpoint, payload=None, data=None, files=None):
        # General method to handle API requests
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        try:
            if files:
                # Handling multipart/form-data for file uploads
                response = self.session.post(url, headers={"Authorization": f"Bearer {self.api_key}"}, data=data, files=files)
            else:
                # Standard JSON-based POST request
                response = self.session.post(url, json=payload, headers=headers)

            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.handle_error(e)

    def handle_error(self, error):
        # Custom error handling logic
        print(f"An error occurred: {error}")
        raise Exception(f"API request error: {error}")
