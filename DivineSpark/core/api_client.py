# DivineSpark/core/api_client.py

class BaseAPIClient:
    def generate_completion(self, prompt, model_name, max_tokens):
        """
        Generates a text completion.

        Args:
            prompt (str): The input prompt.
            model_name (str): The name of the model to use.
            max_tokens (int): Maximum number of tokens to generate.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate_image(self, prompt, model_name):
        """
        Generates an image based on the prompt.

        Args:
            prompt (str): The input prompt.
            model_name (str): The name of the image generation model to use.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def transcribe_audio(self, audio_file, model_name):
        """
        Transcribes the provided audio file.

        Args:
            audio_file (str): The path to the audio file.
            model_name (str): The name of the audio transcription model to use.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
