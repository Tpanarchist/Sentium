import requests
from ..core.api_client import BaseAPIClient

class OpenAIClient(BaseAPIClient):
    def __init__(self, api_key, base_url="https://api.openai.com/v1"):
        super().__init__(api_key, base_url)
        self.session = requests.Session()  # Initialize the session for persistent use
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def load_model(self, model_name, **kwargs):
        print(f"Loading OpenAI model: {model_name}")
        return {"model_name": model_name, **kwargs}

    def generate_completion(self, prompt, model_name="gpt-4o", max_tokens=50):
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        return self.make_request("chat/completions", payload)

    def generate_image(self, prompt, model_name="dall-e-3"):
        payload = {
            "model": model_name,
            "prompt": prompt
        }
        return self.make_request("images/generations", payload)

    def transcribe_audio(self, audio_file_path, model_name="whisper-1"):
        with open(audio_file_path, "rb") as audio_file:
            files = {
                'file': (audio_file_path, audio_file, 'audio/wav')
            }
            data = {
                "model": model_name
            }
            return self.make_request("audio/transcriptions", data=data, files=files)

    def make_request(self, endpoint, payload=None, data=None, files=None):
        url = f"{self.base_url}/{endpoint}"
        try:
            if files:
                # Use multipart/form-data for file uploads
                response = self.session.post(url, data=data, files=files)
            else:
                # Standard JSON POST request
                response = self.session.post(url, json=payload)

            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.handle_error(e)

    def handle_error(self, error):
        error_message = f"API request error: {error}"
        if hasattr(error, 'response') and error.response is not None:
            error_message += f" | Status Code: {error.response.status_code}"
            try:
                error_data = error.response.json()
                error_message += f" | Response: {error_data}"
            except ValueError:
                error_message += f" | Response Text: {error.response.text}"
        raise Exception(error_message)
